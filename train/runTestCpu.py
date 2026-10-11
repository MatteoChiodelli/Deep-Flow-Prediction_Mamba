################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, L. Prantl, X. Hu (TUM)
#
# This file is a copy of runTest.py, which - for convenience - can be run
# on a CPU. This is convenient for quick testing on non-GPU machines.
# If you have a GPU, the runTest.py script is strongly recommended instead.
#
################

import os, sys, random, math
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import TurbDataset
import utils
from utils import log

model_type = sys.argv[1] if len(sys.argv) > 1 else "baseline"
run_name   = sys.argv[2] if len(sys.argv) > 2 else "run"
seed_arg   = int(sys.argv[3]) if len(sys.argv) > 3 else None

if seed_arg is not None:
    seed = seed_arg
else:
    seed = random.randint(0, 2**32 - 1)


prefix = f"{model_type}_{run_name}_s{seed}_"
suffix = ""
print("Output prefix: {}".format(prefix))

expo = 3
dataset = TurbDataset(None, mode=TurbDataset.TEST, dataDirTest="../data/test/")
testLoader = DataLoader(dataset, batch_size=1, shuffle=False)

targets = torch.FloatTensor(1, 3, 128, 128)
targets = Variable(targets)
inputs = torch.FloatTensor(1, 3, 128, 128)
inputs = Variable(inputs)

targets_dn = torch.FloatTensor(1, 3, 128, 128)
targets_dn = Variable(targets_dn)
outputs_dn = torch.FloatTensor(1, 3, 128, 128)
outputs_dn = Variable(outputs_dn)

def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, pad=1, dropout=0.):
    block = nn.Sequential()

    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))

    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=2, mode='bilinear'))
        # reduce kernel size by one for the upsampling (ie decoder part)
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=(size-1), stride=1, padding=pad, bias=True))

    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout > 0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d(dropout, inplace=True))

    return block
    
class DfpNet(nn.Module):
    def __init__(self, channelExponent=6, dropout=0.):
        super(DfpNet, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(3, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer2b= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer3 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer4 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout , size=4 )
        self.layer5 = blockUNet(channels*8, channels*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout , size=2, pad=0)
        self.layer6 = blockUNet(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout , size=2, pad=0)
     
        self.dlayer6 = blockUNet(channels*8, channels*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout , size=2, pad=0)
        self.dlayer5 = blockUNet(channels*16,channels*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout , size=2, pad=0)
        self.dlayer4 = blockUNet(channels*16,channels*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout ) 
        self.dlayer3 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2 = blockUNet(channels*4, channels  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout )

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, 3, 4, 2, 1, bias=True))

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2b= self.layer2b(out2)
        out3 = self.layer3(out2b)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        dout6 = self.dlayer6(out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2b = torch.cat([dout3, out2b], 1)
        dout2b = self.dlayer2b(dout3_out2b)
        dout2b_out2 = torch.cat([dout2b, out2], 1)
        dout2 = self.dlayer2(dout2b_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1

netG = DfpNet(channelExponent=expo)

lf = "./" + prefix + "testout{}.txt".format(suffix) 
utils.makeDirs(["results_test"])

################################################################################
is_small = True  #HERE
suffix_str = "_small" if is_small else ""

metrics_dir = os.path.join("metrics", f"{model_type}_{run_name}_s{seed}{suffix_str}")
os.makedirs(metrics_dir, exist_ok=True)

metrics_file = os.path.join(metrics_dir, "metrics_test.csv")
plot_path    = os.path.join(metrics_dir, "error_plot_test.png")
################################################################################

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

avgLoss = 0.
losses = []
models = []

plot_model_names = []
plot_err_p = []
plot_err_v = []
plot_err_agg = []

for si in range(25):
    s = chr(96 + si)
    if si == 0: 
        s = "" 
    modelFn = "./" + prefix + "modelG{}{}".format(suffix, s)
    if not os.path.isfile(modelFn):
        continue

    models.append(modelFn)
    log(lf, "Loading " + modelFn)
    netG.load_state_dict(torch.load(modelFn))
    log(lf, "Loaded " + modelFn)

    criterionL1 = nn.L1Loss()
    L1val_accum = 0.0
    L1val_dn_accum = 0.0
    lossPer_p_accum = 0.0
    lossPer_v_accum = 0.0
    lossPer_accum = 0.0

    netG.eval()

    for i, data in enumerate(testLoader, 0):
        inputs_cpu, targets_cpu = data
        targets_cpu, inputs_cpu = targets_cpu.float(), inputs_cpu.float()
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

        with torch.no_grad():
            outputs = netG(inputs)
            lossL1 = criterionL1(outputs, targets)
            
        outputs_cpu = outputs.data.cpu().numpy()[0]
        targets_cpu = targets_cpu.cpu().numpy()[0]

        L1val_accum += lossL1.item()

        eps = 1e-8
        lossPer_p = np.sum(np.abs(outputs_cpu[0] - targets_cpu[0])) / max(np.sum(np.abs(targets_cpu[0])), eps)
        lossPer_v = (np.sum(np.abs(outputs_cpu[1] - targets_cpu[1])) + np.sum(np.abs(outputs_cpu[2] - targets_cpu[2]))) / max(
            np.sum(np.abs(targets_cpu[1])) + np.sum(np.abs(targets_cpu[2])), eps
        )
        lossPer = np.sum(np.abs(outputs_cpu - targets_cpu)) / max(np.sum(np.abs(targets_cpu)), eps)

        lossPer_p_accum += lossPer_p.item()
        lossPer_v_accum += lossPer_v.item()
        lossPer_accum   += lossPer.item()

        log(lf, "Test sample %d" % i)
        log(lf, "    pressure:  abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs_cpu[0] - targets_cpu[0])), lossPer_p.item()))
        log(lf, "    velocity:  abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs_cpu[1] - targets_cpu[1])) + np.sum(np.abs(outputs_cpu[2] - targets_cpu[2])), lossPer_v.item()))
        log(lf, "    aggregate: abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs_cpu - targets_cpu)), lossPer.item()))

        input_ndarray = inputs_cpu.cpu().numpy()[0]
        v_norm = (np.max(np.abs(input_ndarray[0, :, :]))**2 + np.max(np.abs(input_ndarray[1, :, :]))**2)**0.5

        outputs_denormalized = dataset.denormalize(outputs_cpu, v_norm)
        targets_denormalized = dataset.denormalize(targets_cpu, v_norm)

        outputs_denormalized_comp = torch.from_numpy(np.array([outputs_denormalized])).float()
        targets_denormalized_comp = torch.from_numpy(np.array([targets_denormalized])).float()

        outputs_dn.data.resize_as_(outputs_denormalized_comp).copy_(outputs_denormalized_comp)
        targets_dn.data.resize_as_(targets_denormalized_comp).copy_(targets_denormalized_comp)

        lossL1_dn = criterionL1(outputs_dn, targets_dn)
        L1val_dn_accum += lossL1_dn.item()

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

################################################################
        if i == 0:
            results_dir = os.path.join(metrics_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            
            img_path = os.path.join(results_dir, "final_test_sample")
            utils.imageOut(img_path, outputs_cpu, targets_cpu, normalize=False, saveMontage=True)
# --------------------------------------------------------------
    log(lf, "\n")
    n_samples = len(testLoader)
    L1val_accum     /= n_samples
    lossPer_p_accum /= n_samples
    lossPer_v_accum /= n_samples
    lossPer_accum   /= n_samples
    L1val_dn_accum  /= n_samples

    log(lf, "Loss percentage (p, v, combined): %f %%    %f %%    %f %% " % (lossPer_p_accum*100, lossPer_v_accum*100, lossPer_accum*100))
    log(lf, "L1 error: %f" % (L1val_accum))
    log(lf, "Denormalized error: %f" % (L1val_dn_accum))
    log(lf, "\n")

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

    avgLoss += lossPer_accum
    losses.append(lossPer_accum)

if len(losses) > 1:
    avgLoss /= len(losses)
    lossStdErr = np.std(losses) / math.sqrt(len(losses))
    log(lf, "Averaged relative error and std dev:   %f , %f " % (avgLoss, lossStdErr))

if len(plot_model_names) > 0:
    x = np.arange(len(plot_model_names))
    width = 0.25

    plt.figure(figsize=(8 + len(plot_model_names), 5))
    plt.bar(x - width, plot_err_p, width, label="Rel. Error Pressure ($p$) %", color="#d62728")
    plt.bar(x, plot_err_v, width, label="Rel. Error Velocity ($v$) %", color="#1f77b4")
    plt.bar(x + width, plot_err_agg, width, label="Aggregate Rel. Error %", color="#2ca02c")

    plt.ylabel("Relative Error (%)", fontsize=11)
    plt.title("TurbNet Baseline - Test Relative Error by Field", fontsize=13)
    plt.xticks(x, plot_model_names, rotation=15 if len(plot_model_names) > 3 else 0)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()

    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\nTest completed.")
    print(f"Metrics saved in: {metrics_file}")
    print(f"Plot saved in:  {plot_path}")