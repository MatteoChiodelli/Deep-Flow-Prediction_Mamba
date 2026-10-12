################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Main training script (Modified with Metrics Logging & Plotting)
#
################

import os, sys, random
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

from DfpNet import TurbNetG, TurbNetAttention, TurbNetMambaBot,TurbNetMambaSup, weights_init
import dataset
import utils


######## Settings ########

# number of training iterations
iterations = 90000
# batch size
batch_size = 10
# learning rate, generator
lrG = 0.0006
# decay learning rate?
decayLr = True
# channel exponent to control network size
expo = 3
# data set config
prop = None
# save txt files with per epoch loss?
saveL1 = False

model_type = sys.argv[1] if len(sys.argv) > 1 else "baseline"
run_name   = sys.argv[2] if len(sys.argv) > 2 else "run"
seed_arg   = int(sys.argv[3]) if len(sys.argv) > 3 else None

if seed_arg is not None:
    seed = seed_arg
else:
    seed = random.randint(0, 2**32 - 1)

prefix = f"{model_type}_{run_name}_s{seed}_"

##########################



dropout = 0.0
doLoad  = ""

print("Model Type: {}".format(model_type))
print("LR: {}".format(lrG))
print("LR decay: {}".format(decayLr))
print("Iterations: {}".format(iterations))
print("Dropout: {}".format(dropout))

##########################

print("Seed: {}".format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Dataset & DataLoader
data = dataset.TurbDataset(prop, shuffle=1)
dataValidation = dataset.ValiDataset(data)

trainLoader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
print("Training batches: {}".format(len(trainLoader)))

valiLoader = DataLoader(dataValidation, batch_size=batch_size, shuffle=False, drop_last=True) 
print("Validation batches: {}".format(len(valiLoader)))

epochs = int(iterations / len(trainLoader) + 0.5)
if epochs < 1:
    epochs = 1

if model_type == "baseline":
    netG = TurbNetG(channelExponent=expo, dropout=dropout)
elif model_type == "attention":
    netG = TurbNetAttention(channelExponent=expo, dropout=dropout)
elif model_type == "mambaBot":
    netG = TurbNetMambaBot(channelExponent=expo, dropout=dropout)
elif model_type == "mambaSup":
    netG = TurbNetMambaSup(channelExponent=expo, dropout=dropout)
else:
    raise ValueError(f"Unknown Model: {model_type}")

print(netG)
model_parameters = filter(lambda p: p.requires_grad, netG.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized {} with {} trainable params".format(model_type, params))

netG.apply(weights_init)
if len(doLoad) > 0:
    netG.load_state_dict(torch.load(doLoad))
    print("Loaded model " + doLoad)

criterionL1 = nn.L1Loss()
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.0)

targets = Variable(torch.FloatTensor(batch_size, 3, 128, 128))
inputs  = Variable(torch.FloatTensor(batch_size, 3, 128, 128))

trainable_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
print(f"Initialized {model_type} with {trainable_params:,} trainable params")

################################################################################

is_small = True  #HERE
suffix_str = "_small" if is_small else ""

metrics_dir = os.path.join("metrics", f"{model_type}_{run_name}_s{seed}{suffix_str}")
os.makedirs(metrics_dir, exist_ok=True)

metrics_file = os.path.join(metrics_dir, "metrics_train.csv")
plot_path = os.path.join(metrics_dir, "loss_curve_train.png")
################################################################################

with open(metrics_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch", 
        "train_loss", 
        "val_loss", 
        "train_rel_error", 
        "val_rel_error", 
        "train_accuracy_pct", 
        "val_accuracy_pct"
    ])

history_train_loss = []
history_val_loss = []

##########################

for epoch in range(epochs):
    print("Starting epoch {} / {}".format((epoch + 1), epochs))

    netG.train()
    L1_accum = 0.0
    train_abs_diff_sum = 0.0
    train_target_sum = 0.0

    for i, traindata in enumerate(trainLoader, 0):
        inputs_cpu, targets_cpu = traindata
        inputs.data.copy_(inputs_cpu.float())
        targets.data.copy_(targets_cpu.float())

        if decayLr:
            currLr = utils.computeLR(epoch, epochs, lrG * 0.1, lrG)
            if currLr < lrG:
                for g in optimizerG.param_groups:
                    g['lr'] = currLr

        netG.zero_grad()
        gen_out = netG(inputs)

        lossL1 = criterionL1(gen_out, targets)
        lossL1.backward()
        optimizerG.step()

        L1_accum += lossL1.item()

        with torch.no_grad():
            train_abs_diff_sum += torch.sum(torch.abs(gen_out - targets)).item()
            train_target_sum += torch.sum(torch.abs(targets)).item()

        if i == len(trainLoader) - 1:
            print("Epoch: {}, batch-idx: {}, L1: {:.6f}".format(epoch, i, lossL1.item()))

    epoch_train_loss = L1_accum / len(trainLoader)
    train_rel_error = train_abs_diff_sum / max(train_target_sum, 1e-8)
    train_acc_pct = max(0.0, (1.0 - train_rel_error) * 100.0)

    netG.eval()
    L1val_accum = 0.0
    val_abs_diff_sum = 0.0
    val_target_sum = 0.0

    with torch.no_grad():
        for i, validata in enumerate(valiLoader, 0):
            inputs_cpu, targets_cpu = validata
            targets_cpu, inputs_cpu = targets_cpu.float(), inputs_cpu.float()
            inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
            targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

            outputs = netG(inputs)
            outputs_cpu = outputs.data.cpu().numpy()
            
            lossL1 = criterionL1(outputs, targets)
            L1val_accum += lossL1.item()
            
            val_abs_diff_sum += torch.sum(torch.abs(outputs - targets)).item()
            val_target_sum += torch.sum(torch.abs(targets)).item()

            if i == 0 and epoch == (epochs - 1):
                input_ndarray = inputs_cpu.cpu().numpy()[0]
                v_norm = ( np.max(np.abs(input_ndarray[0,:,:]))**2 + np.max(np.abs(input_ndarray[1,:,:]))**2 )**0.5

                outputs_denormalized = data.denormalize(outputs_cpu[0], v_norm)
                targets_denormalized = data.denormalize(targets_cpu.cpu().numpy()[0], v_norm)
                
                results_dir = os.path.join(metrics_dir, "results")
                os.makedirs(results_dir, exist_ok=True)
                
                img_path = os.path.join(results_dir, "final_train_sample")
                utils.imageOut(img_path, outputs_denormalized, targets_denormalized, saveTargets=True)
# ------------------------------------------------------------

    epoch_val_loss = L1val_accum / len(valiLoader)
    val_rel_error = val_abs_diff_sum / max(val_target_sum, 1e-8)
    val_acc_pct = max(0.0, (1.0 - val_rel_error) * 100.0)

    print(f"[Epoch {epoch+1:03d}/{epochs:03d}] Train Loss: {epoch_train_loss:.6f} (Acc: {train_acc_pct:.2f}%) | Val Loss: {epoch_val_loss:.6f} (Acc: {val_acc_pct:.2f}%)")

    with open(metrics_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch + 1, 
            epoch_train_loss, 
            epoch_val_loss, 
            train_rel_error, 
            val_rel_error, 
            train_acc_pct, 
            val_acc_pct
        ])

    history_train_loss.append(epoch_train_loss)
    history_val_loss.append(epoch_val_loss)

torch.save(netG.state_dict(), prefix + "modelG")

plt.figure(figsize=(9, 5))
plt.plot(range(1, epochs + 1), history_train_loss, label="Train Loss (L1)", color="#1f77b4", linewidth=2)
plt.plot(range(1, epochs + 1), history_val_loss, label="Validation Loss (L1)", color="#ff7f0e", linestyle="--", linewidth=2)
plt.title("TurbNet Baseline - Training & Validation Loss Over Time", fontsize=13)
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