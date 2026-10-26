################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Main training script
#
# Modified by our group w.r.t. the original DFP repo:
#  - the model is chosen by name (build_model in DfpNet.py) and passed on the command line,
#    together with run name, seed and optional V-Mamba hyperparameters (see run_all_seeds.py / runTuning.py)
#  - fixed seeds, optional fixed train/val split, mixed precision (AMP), more DataLoader workers
#  - per-epoch metrics (relative errors per field, time, peak VRAM) saved to CSV + loss plot in metrics/
#
################

import os, sys, random, time
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from DfpNet import build_model, weights_init
import dataset
import utils
import torch.backends.cudnn as cudnn
######## Settings ########

cudnn.enabled = True
cudnn.benchmark = True

# number of training iterations
iterations = 400000
# batch size
batch_size = 10
# learning rate, generator
lrG = 0.0006
# decay learning rate?
decayLr = True
# channel exponent to control network size (runTuning.py passes it via DFP_EXPO)
expo = int(os.environ.get("DFP_EXPO", 5))
# data set config: [total samples, fraction regular, fraction superimposed, fraction sheared]
prop = [53830, 0.4964, 0.0, 0.5036]
# save txt files with per epoch loss?
saveL1 = False
# fixed seed for the train/val split, so that all seeds are validated on the same samples.
# None = split depends on the run seed (original behavior, needed to compare with previous runs);
# 42 reproduces the split of the previous seed-42 runs for every seed
split_seed = None

# Command line (called by run_all_seeds.py and runTuning.py):
#   python runTrain.py <model_type> <run_name> <seed> [<tuning_dir>|default] [9 or 12 V-Mamba hyperparameters]
# model_type is a key of MODEL_CONFIGS in DfpNet.py. The optional hyperparameters are
# (ssm_d_state, ssm_ratio, mlp_ratio) for vmamba1, vmamba2, vmamba3 and, with 12 values, vmamba_decoder.
model_type = sys.argv[1] if len(sys.argv) > 1 else "baseline"
run_name   = sys.argv[2] if len(sys.argv) > 2 else "run"
seed_arg   = int(sys.argv[3]) if len(sys.argv) > 3 else None
tuning_dir = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "default" else None

mamba_cfg = None
extra_args = sys.argv[5:]
if len(extra_args) in (9, 12):
    vals = [float(x) if '.' in x else int(x) for x in extra_args]
    mamba_cfg = {
        "vmamba1": {"ssm_d_state": vals[0], "ssm_ratio": vals[1], "mlp_ratio": vals[2]},
        "vmamba2": {"ssm_d_state": vals[3], "ssm_ratio": vals[4], "mlp_ratio": vals[5]},
        "vmamba3": {"ssm_d_state": vals[6], "ssm_ratio": vals[7], "mlp_ratio": vals[8]},
    }
    if len(vals) == 12:
        mamba_cfg["vmamba_decoder"] = {"ssm_d_state": vals[9], "ssm_ratio": vals[10], "mlp_ratio": vals[11]}
elif len(extra_args) > 0:
    raise ValueError(f"Pass 9 or 12 hyperparametrers: {len(extra_args)}")

# fixed seed if given (we always use 42, 1955, 2003), random otherwise as in the original script
if seed_arg is not None:
    seed = seed_arg
else:
    seed = random.randint(0, 2**32 - 1)

prefix = f"{model_type}_{run_name}_s{seed}_"
dropout = 0.0
doLoad  = ""

print("Model Type: {}".format(model_type))
print("LR: {}".format(lrG))
print("LR decay: {}".format(decayLr))
print("Iterations: {}".format(iterations))
print("Dropout: {}".format(dropout))
print("Channel exponent: {}".format(expo))
print("Split seed: {}".format(split_seed))
##########################

print("Random seed: {}".format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# create pytorch data object with dfp dataset
data = dataset.TurbDataset(prop, shuffle=1, splitSeed=split_seed)
trainLoader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
print("Training batches: {}".format(len(trainLoader)))
dataValidation = dataset.ValiDataset(data)
valiLoader = DataLoader(dataValidation, batch_size=batch_size, shuffle=False, drop_last=False,num_workers=4,pin_memory=True) 
print("Validation batches: {}".format(len(valiLoader)))

# setup training
epochs = int(iterations/len(trainLoader) + 0.5)

if epochs < 1:
    epochs = 1

# originally TurbNetG(...): now the network is built from its name (baseline or one of our variants)
netG = build_model(model_type, channelExponent=expo, dropout=dropout, mamba_cfg=mamba_cfg)

print(netG) # print full net
model_parameters = filter(lambda p: p.requires_grad, netG.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized TurbNet with {} trainable params ".format(params))

netG.apply(weights_init)
if len(doLoad)>0:
    netG.load_state_dict(torch.load(doLoad))
    print("Loaded model "+doLoad)
netG.cuda()

criterionL1 = nn.L1Loss()
criterionL1.cuda()

trainable_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
print(f"Initialized {model_type} with {trainable_params:,} trainable params")

optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.0)
# added: mixed precision training to speed up the V-Mamba runs
scaler = torch.cuda.amp.GradScaler()

targets = Variable(torch.FloatTensor(batch_size, 3, 128, 128))
inputs  = Variable(torch.FloatTensor(batch_size, 3, 128, 128))
targets = targets.cuda()
inputs  = inputs.cuda()

# Output folder (added): metrics/<model>/<model>_<run>_s<seed>[_small]/, or metrics/<tuning_dir>/<run>/
# when called by runTuning.py. "_small" marks runs on the reduced dataset (data/train_small).
is_small = "small" in os.path.basename(os.path.normpath(data.dataDir))
suffix_str = "_small" if is_small else ""

if tuning_dir:
    metrics_dir = os.path.join("metrics", tuning_dir, run_name)
else:
    metrics_dir = os.path.join("metrics", model_type, f"{model_type}_{run_name}_s{seed}{suffix_str}")

os.makedirs(metrics_dir, exist_ok=True)

metrics_file = os.path.join(metrics_dir, "metrics_train.csv")
plot_path = os.path.join(metrics_dir, "loss_curve_train.png")

# one CSV row per epoch. Relative error = sum|pred - target| / sum|target| (same definition as the
# original runTest.py), "accuracy" = 1 - relative error
with open(metrics_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch", 
        "train_loss", 
        "val_loss", 
        "train_rel_error", 
        "val_rel_error", 
        "train_accuracy_pct", 
        "val_accuracy_pct",
        "val_err_p",   
        "val_err_vx",  
        "val_err_vy",
        "epoch_time_s",
        "peak_vram_mb"
    ])

history_train_loss = []
history_val_loss = []
##########################

for epoch in range(epochs):
    print("Starting epoch {} / {}".format((epoch+1),epochs))

    epoch_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    netG.train()
    L1_accum = 0.0

    # added: accumulators for the training relative error
    train_abs_diff_sum = 0.0
    train_target_sum = 0.0
    for i, traindata in enumerate(trainLoader, 0):
        inputs_cpu, targets_cpu = traindata
        targets_cpu, inputs_cpu = targets_cpu.float().cuda(), inputs_cpu.float().cuda()
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

        # compute LR decay
        if decayLr:
            currLr = utils.computeLR(epoch, epochs, lrG*0.1, lrG)
            if currLr < lrG:
                for g in optimizerG.param_groups:
                    g['lr'] = currLr

        netG.zero_grad()
        with torch.cuda.amp.autocast():
            gen_out = netG(inputs)
            lossL1 = criterionL1(gen_out, targets)

        scaler.scale(lossL1).backward()

        scaler.step(optimizerG)
        scaler.update()

        lossL1viz = lossL1.item()
        L1_accum += lossL1viz

        with torch.no_grad():
            train_abs_diff_sum += torch.sum(torch.abs(gen_out - targets))
            train_target_sum += torch.sum(torch.abs(targets))
        if i==len(trainLoader)-1:
            logline = "Epoch: {}, batch-idx: {}, L1: {}\n".format(epoch, i, lossL1viz)
            print(logline)

    epoch_train_loss = L1_accum / len(trainLoader)
    train_rel_error = train_abs_diff_sum.item() / max(train_target_sum.item(), 1e-8)
    train_acc_pct = max(0.0, (1.0 - train_rel_error) * 100.0)
    # validation
    netG.eval()
    L1val_accum = 0.0

    # added: accumulators for the validation relative error, total and per field (p, vx, vy)
    val_abs_diff_sum = 0.0
    val_target_sum = 0.0
    val_abs_diff_p = 0.0
    val_target_p = 0.0
    val_abs_diff_vx = 0.0
    val_target_vx = 0.0
    val_abs_diff_vy = 0.0
    val_target_vy = 0.0

    with torch.no_grad():
        for i, validata in enumerate(valiLoader, 0):
            inputs_cpu, targets_cpu = validata
            targets_cpu, inputs_cpu = targets_cpu.float().cuda(), inputs_cpu.float().cuda()
            inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
            targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

            outputs = netG(inputs)

            lossL1 = criterionL1(outputs, targets)
            L1val_accum += lossL1.item() * inputs.size(0)
            
            val_abs_diff_sum += torch.sum(torch.abs(outputs - targets)).item()
            val_target_sum += torch.sum(torch.abs(targets)).item()

            val_abs_diff_p += torch.sum(torch.abs(outputs[:, 0, :, :] - targets[:, 0, :, :])).item()
            val_target_p += torch.sum(torch.abs(targets[:, 0, :, :])).item()
            
            val_abs_diff_vx += torch.sum(torch.abs(outputs[:, 1, :, :] - targets[:, 1, :, :])).item()
            val_target_vx += torch.sum(torch.abs(targets[:, 1, :, :])).item()
            
            val_abs_diff_vy += torch.sum(torch.abs(outputs[:, 2, :, :] - targets[:, 2, :, :])).item()
            val_target_vy += torch.sum(torch.abs(targets[:, 2, :, :])).item()

            # changed: only one sample image at the last epoch, saved in the run folder
            if i == 0 and epoch == (epochs - 1):
                outputs_cpu = outputs.data.cpu().numpy()
                input_ndarray = inputs_cpu.cpu().numpy()[0]
                # changed: v_norm must come from the raw (un-normalized) inflow velocity, the original
                # code used the normalized one
                vx_max = np.max(np.abs(input_ndarray[0,:,:])) * data.max_inputs_0
                vy_max = np.max(np.abs(input_ndarray[1,:,:])) * data.max_inputs_1
                v_norm = ( vx_max**2 + vy_max**2 )**0.5

                outputs_denormalized = data.denormalize(outputs_cpu[0], v_norm)
                targets_denormalized = data.denormalize(targets_cpu.cpu().numpy()[0], v_norm)
                
                results_dir = os.path.join(metrics_dir, "results")
                os.makedirs(results_dir, exist_ok=True)
                
                img_path = os.path.join(results_dir, "final_train_sample")
                utils.imageOut(img_path, outputs_denormalized, targets_denormalized, saveTargets=True)

    # added: epoch time and peak GPU memory, used to compare the cost of the different models
    torch.cuda.synchronize()
    epoch_time_s = time.perf_counter() - epoch_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    epoch_val_loss = L1val_accum / len(dataValidation)
    val_rel_error = val_abs_diff_sum / max(val_target_sum, 1e-8)
    val_acc_pct = max(0.0, (1.0 - val_rel_error) * 100.0)

    val_err_p_pct = (val_abs_diff_p / max(val_target_p, 1e-8)) * 100.0
    val_err_vx_pct = (val_abs_diff_vx / max(val_target_vx, 1e-8)) * 100.0
    val_err_vy_pct = (val_abs_diff_vy / max(val_target_vy, 1e-8)) * 100.0

    print(f"[Epoch {epoch+1:03d}/{epochs:03d}] Train Loss: {epoch_train_loss:.6f} (Acc: {train_acc_pct:.2f}%) | Val Loss: {epoch_val_loss:.6f} (Acc: {val_acc_pct:.2f}%)")
    print(f"                                   -> Val Error p: {val_err_p_pct:.2f}%, vx: {val_err_vx_pct:.2f}%, vy: {val_err_vy_pct:.2f}%")
    print(f"                                   -> Time: {epoch_time_s:.1f}s, Peak VRAM: {peak_vram_mb:.0f} MB")

    with open(metrics_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch + 1, 
            epoch_train_loss, 
            epoch_val_loss, 
            train_rel_error, 
            val_rel_error, 
            train_acc_pct, 
            val_acc_pct,

            val_err_p_pct,  
            val_err_vx_pct, 
            val_err_vy_pct,
            epoch_time_s,
            peak_vram_mb
        ])

    history_train_loss.append(epoch_train_loss)
    history_val_loss.append(epoch_val_loss)

# changed: weights are saved in the run folder instead of the working directory
torch.save(netG.state_dict(), os.path.join(metrics_dir, prefix + "modelG"))

# added: train/validation loss curve
plt.figure(figsize=(9, 5))
plt.plot(range(1, epochs + 1), history_train_loss, label="Train Loss (L1)", color="#1f77b4", linewidth=2)
plt.plot(range(1, epochs + 1), history_val_loss, label="Validation Loss (L1)", color="#ff7f0e", linestyle="--", linewidth=2)
plt.title(f"{model_type.capitalize()} - Training & Validation Loss", fontsize=13)
plt.xlabel("Epochs", fontsize=11)
plt.ylabel("L1 Loss", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

plt.savefig(plot_path, dpi=300)
plt.close()

print(f"\nTraining completed.")
print(f"Metrics saved in: {metrics_file}")
print(f"Plot saved in:  {plot_path}")
