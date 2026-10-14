################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# CNN setup and data normalization
#
################


import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

###############
# 
# Modifications for introduction of self attention / mamba blocks
# Adaptive based on local/gpu requirements.txt

# Checking Mamba standard (1D) GPU
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

# Checking V-Mamba (Selective Scan 2D) GPU
try:
    # Importive selective_scan from MambaV repo for 2D scanning
    from vmamba import VSSBlock  
    HAS_VMAMBA = True
except ImportError:
    HAS_VMAMBA = False

if torch.cuda.is_available():
    print(f"[INFO] CUDA avaible. Mamba-1D: {HAS_MAMBA} | V-Mamba (SS2D): {HAS_VMAMBA}")
    if HAS_VMAMBA:
        # vmamba silently falls back to slow PyTorch kernels when the CUDA/Triton ones are missing
        import os, shutil
        import vmamba
        scan_backend = "CUDA (mamba_ssm)" if vmamba.WITH_SELECTIVESCAN_MAMBA else "PyTorch fallback (slow)"
        cross_backend = "Triton" if vmamba.WITH_TRITON else "PyTorch fallback (slow)"
        # Triton compiles its kernels at runtime and needs a C compiler (same lookup as triton/common/build.py)
        if vmamba.WITH_TRITON and not (os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")):
            cross_backend += " - ERROR: no C compiler found (install gcc), V-Mamba forward will fail"
        print(f"[INFO] V-Mamba selective scan: {scan_backend} | cross-scan: {cross_backend}")
else:
    print("[WARNING] CUDA not avaible. Mamba blocks forced on CPU.")
#################



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, pad=1, dropout=0.):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=2, mode='bilinear')) # Note: old default was nearest neighbor
        # reduce kernel size by one for the upsampling (ie decoder part)
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=(size-1), stride=1, padding=pad, bias=True))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout>0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d( dropout, inplace=True))
    return block
    
# generator model
class TurbNetG(nn.Module):
    def __init__(self, channelExponent=6, dropout=0.):
        super(TurbNetG, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(3, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer2b= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer3 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout )
        # note the following layer also had a kernel size of 2 in the original version (cf https://arxiv.org/abs/1810.08217)
        # it is now changed to size 4 for encoder/decoder symmetry; to reproduce the old/original results, please change it to 2
        self.layer4 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout ,  size=4 ) # note, size 4!
        self.layer5 = blockUNet(channels*8, channels*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout , size=2,pad=0)
        self.layer6 = blockUNet(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout , size=2,pad=0)
     
        # note, kernel size is internally reduced by one now
        self.dlayer6 = blockUNet(channels*8, channels*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer5 = blockUNet(channels*16,channels*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
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

# discriminator (only for adversarial training, currently unused)
class TurbNetD(nn.Module):
    def __init__(self, in_channels1, in_channels2,ch=64):
        super(TurbNetD, self).__init__()

        self.c0 = nn.Conv2d(in_channels1 + in_channels2, ch, 4, stride=2, padding=2)
        self.c1 = nn.Conv2d(ch  , ch*2, 4, stride=2, padding=2)
        self.c2 = nn.Conv2d(ch*2, ch*4, 4, stride=2, padding=2)
        self.c3 = nn.Conv2d(ch*4, ch*8, 4, stride=2, padding=2)
        self.c4 = nn.Conv2d(ch*8, 1   , 4, stride=2, padding=2)

        self.bnc1 = nn.BatchNorm2d(ch*2)
        self.bnc2 = nn.BatchNorm2d(ch*4)
        self.bnc3 = nn.BatchNorm2d(ch*8)        

    def forward(self, x1, x2):
        h = self.c0(torch.cat((x1, x2),1))
        h = self.bnc1(self.c1(F.leaky_relu(h, negative_slope=0.2)))
        h = self.bnc2(self.c2(F.leaky_relu(h, negative_slope=0.2)))
        h = self.bnc3(self.c3(F.leaky_relu(h, negative_slope=0.2)))
        h = self.c4(F.leaky_relu(h, negative_slope=0.2))
        h = F.sigmoid(h) 
        return h


# Novelty ----------------------------------------------------------------------------------------------------------------------------------------

class VMamba_Block(nn.Module):
    def __init__(self, in_channels,ssm_d_state=16,ssm_ratio=2,ssm_conv=2,mlp_ratio=4):
        super().__init__()
        self.vss_block = VSSBlock(hidden_dim=in_channels, channel_first=True,ssm_d_state=ssm_d_state,ssm_ratio=ssm_ratio,ssm_conv=ssm_conv,mlp_ratio=mlp_ratio,forward_type="v02")   
#hidden dim = input (input is equal to output) 
#ssm_d_state = how big the hidden state is (defaul=16)
#ssm_ratio = how much i increase the size of the input (default=2)
#channel_first= assumes that the first dimension of the tensor is the channel (for us ALWAYS True)
#ssm_conv= kernel of the local 1d convolution used to catch close relations (default=3)
#mlp_ratio= how much "bigger" is the hidden state of the final mlp (default=4)

    def forward(self, x):
        out = self.vss_block(x.contiguous())
        return out


class MultiHeadSelfAttention2d(nn.Module):
    #Multi-head self-attention applied to a 2D feature map

    def __init__(self, channels, num_heads=8, dropout=0.):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        # Convert [B, C, H, W] to [B, H*W, C], as expected by attention.
        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normalized_tokens = self.norm(tokens)
        attended_tokens, _ = self.attention(
            normalized_tokens,
            normalized_tokens,
            normalized_tokens,
            need_weights=False,
        )

        attended = attended_tokens.transpose(1, 2).reshape(
            batch_size, channels, height, width
        )
        return x + attended


# Configurable TurbNet ---------------------------------------------------------------------------------------------------------------------------
#
# Same U-Net as TurbNetG, plus optional "mixer" blocks (V-Mamba or self-attention) applied to the output of encoder stages.
# A mixer replaces the stage output both for the next encoder layer and for the skip connection.
#
# Encoder stage outputs for a 128x128 input (channels as multiples of 2**channelExponent):
#   in   128x128, 3 channels (network input, before layer1)
#   out1  64x64 x1 | out2  32x32 x2 | out2b 16x16 x2 | out3  8x8 x4 | out4  4x4 x8 | out5  2x2 x8 | out6  1x1 x8
INPUT_CHANNELS = 3
STAGE_CHANNELS = {"in": None, "out1": 1, "out2": 2, "out2b": 2, "out3": 4, "out4": 8, "out5": 8, "out6": 8}

# model name -> {stage: mixer type}
MODEL_CONFIGS = {
    # Original variants, kept identical to reproduce previous runs and load their checkpoints
    "baseline":  {},
    "mambaSup":  {"out2b": "vmamba_big"},
    "fullMamba": { "out1": "vmamba_big", "out2_big": "vmamba_big", "out2b": "vmamba_small", "out3": "vmamba_small", "out4": "vmamba_small", "out5": "vmamba_small"},
    # Experimental configurations (README, section 2)
    "C0": {},                                          # baseline
    "C1": {"out1": "vmamba_big", "out4": "attention"},     # high-res Mamba + bottleneck attention
    "C2": {"out1": "vmamba_big"},                          # high-res Mamba only
    "C3": {"out4": "vmamba_small"},                          # bottleneck Mamba only
    "C4": {"out4": "attention"},                       # bottleneck attention only
}


class TurbNet(nn.Module):
    def __init__(self, mixers=None, channelExponent=6, dropout=0., num_heads=8, attention_dropout=0.):
        super(TurbNet, self).__init__()
        channels = int(2 ** channelExponent + 0.5)
        mixers = dict(mixers or {})
        unknown = set(mixers) - set(STAGE_CHANNELS)
        if unknown:
            raise ValueError(f"Unknown stages: {sorted(unknown)}. Available: {list(STAGE_CHANNELS)}")
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout

        # module name per stage: "layer_<type>" (as in the original variants), "layer_<type>_<stage>" if the type is used more than once
        kinds = list(mixers.values())
        self.mixer_names = {
            stage: f"layer_{kind}" if kinds.count(kind) == 1 else f"layer_{kind}_{stage}"
            for stage, kind in mixers.items()
        }
        self._mixers = mixers
        self._channels = channels

        # modules are registered in forward order (mixers right after their stage) so that, with the same seed,
        # the weight initialization matches the original TurbNetG / TurbNetMambaSup / TurbNetMambaBot / TurbNetAttention
        self._add_mixer("in")
        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(3, channels, 4, 2, 1, bias=True))
        self._add_mixer("out1")

        self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout )
        self._add_mixer("out2")
        self.layer2b= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout )
        self._add_mixer("out2b")
        self.layer3 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout )
        self._add_mixer("out3")
        self.layer4 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout ,  size=4 ) # note, size 4!
        self._add_mixer("out4")
        self.layer5 = blockUNet(channels*8, channels*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout , size=2,pad=0)
        self._add_mixer("out5")
        self.layer6 = blockUNet(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout , size=2,pad=0)
        self._add_mixer("out6")

        self.dlayer6 = blockUNet(channels*8, channels*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer5 = blockUNet(channels*16,channels*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer4 = blockUNet(channels*16,channels*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout ) 
        self.dlayer3 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2 = blockUNet(channels*4, channels  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout )

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, 3, 4, 2, 1, bias=True))

    def _add_mixer(self, stage):
        kind = self._mixers.get(stage)
        if kind is None:
            return
        in_channels = INPUT_CHANNELS if stage == "in" else self._channels * STAGE_CHANNELS[stage]
        if kind == "vmamba_big":
            block = VMamba_Block(in_channels=in_channels,)
        elif kind=="vmamba_small":
            block = VMamba_Block(in_channels=in_channels,mlp_ratio=2,ssm_ratio=1,ssm_d_state=8)
        elif kind == "attention":
            block = MultiHeadSelfAttention2d(channels=in_channels, num_heads=self.num_heads, dropout=self.attention_dropout)
        else:
            raise ValueError(f"Unknown mixer type: {kind}")
        self.add_module(self.mixer_names[stage], block)

    def _mix(self, stage, x):
        name = self.mixer_names.get(stage)
        return x if name is None else getattr(self, name)(x)

    def forward(self, x):
        x    = self._mix("in",    x)
        out1 = self._mix("out1",  self.layer1(x))
        out2 = self._mix("out2",  self.layer2(out1))
        out2b= self._mix("out2b", self.layer2b(out2))
        out3 = self._mix("out3",  self.layer3(out2b))
        out4 = self._mix("out4",  self.layer4(out3))
        out5 = self._mix("out5",  self.layer5(out4))
        out6 = self._mix("out6",  self.layer6(out5))
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


def build_model(model_type, channelExponent=6, dropout=0.):
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown Model: {model_type}. Available: {list(MODEL_CONFIGS)}")
    return TurbNet(MODEL_CONFIGS[model_type], channelExponent=channelExponent, dropout=dropout)
