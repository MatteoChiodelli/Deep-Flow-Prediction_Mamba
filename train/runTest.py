################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Compute errors for a test set and visualize. This script can loop over a range of models in 
# order to compute an averaged evaluation. 
#
################

import os,sys,random,math
import numpy as np
########################################################################################################            val_abs_diff_sum += torch.sum(torch.abs(outputs - targets)).item()
import csv
import matplotlib.pyplot as plt
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import TurbDataset
from DfpNet import build_model, weights_init

import utils
from utils import log

########################################################################################################            val_abs_diff_sum += torch.sum(torch.abs(outputs - targets)).item()
model_type = sys.argv[1] if len(sys.argv) > 1 else "baseline"
run_name   = sys.argv[2] if len(sys.argv) > 2 else "run"
seed_arg   = int(sys.argv[3]) if len(sys.argv) > 3 else None
# same Mamba hyperparameters used in training (argv[5:]), otherwise the weights don't match
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

if seed_arg is not None:
    seed = seed_arg
else:
    seed = random.randint(0, 2**32 - 1)

prefix = f"{model_type}_{run_name}_s{seed}_"
suffix = ""
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
expo = 5
dataset = TurbDataset(None, mode=TurbDataset.TEST, dataDirTest="../data/test/")
testLoader = DataLoader(dataset, batch_size=1, shuffle=False)

targets = torch.FloatTensor(1, 3, 128, 128)
targets = Variable(targets)
targets = targets.cuda()
inputs = torch.FloatTensor(1, 3, 128, 128)
inputs = Variable(inputs)
inputs = inputs.cuda()

targets_dn = torch.FloatTensor(1, 3, 128, 128)
targets_dn = Variable(targets_dn)
targets_dn = targets_dn.cuda()
outputs_dn = torch.FloatTensor(1, 3, 128, 128)
outputs_dn = Variable(outputs_dn)
outputs_dn = outputs_dn.cuda()

netG = build_model(model_type, channelExponent=expo, mamba_cfg=mamba_cfg)
lf = "./" + prefix + "testout{}.txt".format(suffix) 
utils.makeDirs(["results_test"])

#################################################################################
is_small = "small" in os.path.basename(os.path.normpath(dataset.dataDir))
suffix_str = "_small" if is_small else ""

metrics_dir = os.path.join("metrics", model_type, f"{model_type}_{run_name}_s{seed}{suffix_str}")
os.makedirs(metrics_dir, exist_ok=True)

metrics_file = os.path.join(metrics_dir, "metrics_test.csv")
plot_path    = os.path.join(metrics_dir, "error_plot_test.png")
################################################################################

if not os.path.isfile(metrics_file):
    with open(metrics_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "sample_id",
            "l1_loss",
            "l1_loss_denormalized",
            "rel_error_pressure_pct",
            "rel_error_velocity_pct",
            "rel_error_aggregate_pct",
            "accuracy_pressure_pct",
            "accuracy_velocity_pct",
            "accuracy_aggregate_pct"
        ])

plot_model_names = []
plot_err_p = []
plot_err_v = []
plot_err_agg = []
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# loop over different trained models
avgLoss = 0.
losses = []
models = []

for si in range(25):
    s = chr(96+si)
    if(si==0): 
        s = "" # check modelG, and modelG + char
    modelFn = "./" + prefix + "modelG{}{}".format(suffix,s)
    if not os.path.isfile(modelFn):
        continue

    models.append(modelFn)
    log(lf, "Loading " + modelFn )
    netG.load_state_dict( torch.load(modelFn) )
    log(lf, "Loaded " + modelFn )
    netG.cuda()

    criterionL1 = nn.L1Loss()
    criterionL1.cuda()
    L1val_accum = 0.0
    L1val_dn_accum = 0.0
    lossPer_p_accum = 0
    lossPer_v_accum = 0
    lossPer_accum = 0

    netG.eval()

    for i, data in enumerate(testLoader, 0):
        inputs_cpu, targets_cpu = data
        targets_cpu, inputs_cpu = targets_cpu.float().cuda(), inputs_cpu.float().cuda()
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

########################################################################################################            val_abs_diff_sum += torch.sum(torch.abs(outputs - targets)).item()
        with torch.no_grad():
            outputs = netG(inputs)
            lossL1 = criterionL1(outputs, targets)
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////        
        outputs_cpu = outputs.data.cpu().numpy()[0]
        targets_cpu = targets_cpu.cpu().numpy()[0]

########################################################################################################            val_abs_diff_sum += torch.sum(torch.abs(outputs - targets)).item()
        L1val_accum += lossL1.item()

        eps = 1e-8
        lossPer_p = np.sum(np.abs(outputs_cpu[0] - targets_cpu[0])) / max(np.sum(np.abs(targets_cpu[0])), eps)
        lossPer_v = (np.sum(np.abs(outputs_cpu[1] - targets_cpu[1])) + np.sum(np.abs(outputs_cpu[2] - targets_cpu[2]))) / max(
            np.sum(np.abs(targets_cpu[1])) + np.sum(np.abs(targets_cpu[2])), eps
        )
        lossPer = np.sum(np.abs(outputs_cpu - targets_cpu)) / max(np.sum(np.abs(targets_cpu)), eps)

        lossPer_p_accum += lossPer_p.item()
        lossPer_v_accum += lossPer_v.item()
        lossPer_accum += lossPer.item()
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        log(lf, "Test sample %d"% i )
        log(lf, "    pressure:  abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs_cpu[0] - targets_cpu[0])), lossPer_p.item()) )
        log(lf, "    velocity:  abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs_cpu[1] - targets_cpu[1])) + np.sum(np.abs(outputs_cpu[2] - targets_cpu[2])) , lossPer_v.item() ) )
        log(lf, "    aggregate: abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs_cpu    - targets_cpu   )), lossPer.item()) )

        # Calculate the norm
        input_ndarray = inputs_cpu.cpu().numpy()[0]
        # NOTE: v_norm is computed on the normalized inputs (original DFP code), so l1_loss_denormalized is not
        # in physical units; kept as is to stay comparable with previous test results
        v_norm = ( np.max(np.abs(input_ndarray[0,:,:]))**2 + np.max(np.abs(input_ndarray[1,:,:]))**2 )**0.5

        outputs_denormalized = dataset.denormalize(outputs_cpu, v_norm)
        targets_denormalized = dataset.denormalize(targets_cpu, v_norm)

        # denormalized error 
        outputs_denormalized_comp=np.array([outputs_denormalized])
        outputs_denormalized_comp=torch.from_numpy(outputs_denormalized_comp)
        targets_denormalized_comp=np.array([targets_denormalized])
        targets_denormalized_comp=torch.from_numpy(targets_denormalized_comp)

        targets_denormalized_comp, outputs_denormalized_comp = targets_denormalized_comp.float().cuda(), outputs_denormalized_comp.float().cuda()

        outputs_dn.data.resize_as_(outputs_denormalized_comp).copy_(outputs_denormalized_comp)
        targets_dn.data.resize_as_(targets_denormalized_comp).copy_(targets_denormalized_comp)

        lossL1_dn = criterionL1(outputs_dn, targets_dn)
        L1val_dn_accum += lossL1_dn.item()

########################################################################################################            val_abs_diff_sum += torch.sum(torch.abs(outputs - targets)).item()
        with open(metrics_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                os.path.basename(modelFn),
                i,
                lossL1.item(),
                lossL1_dn.item(),
                lossPer_p.item() * 100.0,
                lossPer_v.item() * 100.0,
                lossPer.item() * 100.0,
                max(0.0, (1.0 - lossPer_p.item()) * 100.0),
                max(0.0, (1.0 - lossPer_v.item()) * 100.0),
                max(0.0, (1.0 - lossPer.item()) * 100.0)
            ])
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if i == 0:
            results_dir = os.path.join(metrics_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            
            img_path = os.path.join(results_dir, "final_test_sample")
            utils.imageOut(img_path, outputs_cpu, targets_cpu, normalize=False, saveMontage=True)
# --------------------------------------------------------------
    log(lf, "\n") 
    L1val_accum     /= len(testLoader)
    lossPer_p_accum /= len(testLoader)
    lossPer_v_accum /= len(testLoader)
    lossPer_accum   /= len(testLoader)
    L1val_dn_accum  /= len(testLoader)
    
    log(lf, "Loss percentage (p, v, combined): %f %%    %f %%    %f %% " % (lossPer_p_accum*100, lossPer_v_accum*100, lossPer_accum*100 ) )
    log(lf, "L1 error: %f" % (L1val_accum) )
    log(lf, "Denormalized error: %f" % (L1val_dn_accum) )
    log(lf, "\n") 

########################################################################################################            val_abs_diff_sum += torch.sum(torch.abs(outputs - targets)).item()
    with open(metrics_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            os.path.basename(modelFn) + "_AVERAGE",
            "ALL",
            L1val_accum,
            L1val_dn_accum,
            lossPer_p_accum * 100.0,
            lossPer_v_accum * 100.0,
            lossPer_accum * 100.0,
            max(0.0, (1.0 - lossPer_p_accum) * 100.0),
            max(0.0, (1.0 - lossPer_v_accum) * 100.0),
            max(0.0, (1.0 - lossPer_accum) * 100.0)
        ])

    plot_model_names.append(os.path.basename(modelFn))
    plot_err_p.append(lossPer_p_accum * 100.0)
    plot_err_v.append(lossPer_v_accum * 100.0)
    plot_err_agg.append(lossPer_accum * 100.0)
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    avgLoss += lossPer_accum
    losses.append(lossPer_accum)

if len(losses)>1:
    avgLoss /= len(losses)
    lossStdErr = np.std(losses) / math.sqrt(len(losses))
    log(lf, "Averaged relative error and std dev across models:   %f , %f " % (avgLoss,lossStdErr) )

########################################################################################################            val_abs_diff_sum += torch.sum(torch.abs(outputs - targets)).item()
if len(plot_model_names) > 0:
    x = np.arange(len(plot_model_names))
    width = 0.25

    plt.figure(figsize=(8 + len(plot_model_names), 5))
    plt.bar(x - width, plot_err_p, width, label="Rel. Error Pressure ($p$) %", color="#d62728")
    plt.bar(x, plot_err_v, width, label="Rel. Error Velocity ($v$) %", color="#1f77b4")
    plt.bar(x + width, plot_err_agg, width, label="Aggregate Rel. Error %", color="#2ca02c")

    plt.ylabel("Relative Error (%)", fontsize=11)
    plt.title(f"{model_type.capitalize()} - Test Relative Error by Field", fontsize=13)
    plt.xticks(x, plot_model_names, rotation=15 if len(plot_model_names) > 3 else 0)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()

    plot_path = os.path.join(metrics_dir, f"test_errors_{model_type}_{run_name}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\nTest completato.")
    print(f"Metriche salvate in: {metrics_file}")
    print(f"Grafico salvato in:  {plot_path}")
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////