# Mamba-Augmented U-Nets for Fast RANS Flow Prediction Around Airfoils

Code of the project report of Group 61, Deep Machine Learning (Chalmers): Ignazio Spaccavento, Matteo Chiodelli.

We add VMamba blocks (2D selective scan, SS2D) and, for comparison, multi-head self-attention blocks to the
U-Net (TurbNet) of Thuerey et al., *Deep learning methods for Reynolds-averaged Navier-Stokes simulations of
airfoil flows* (AIAA J., 2020). The code builds on their public repository
(<https://github.com/thunil/Deep-Flow-Prediction>, Apache 2.0, see `LICENSE`); the `VSSBlock` comes from the
official VMamba repository (<https://github.com/MzeroMiko/VMamba>).

## Files

All the code is in `train/` and must be run from that folder.

| File | Origin | Content |
|---|---|---|
| `DfpNet.py` | original + ours | Original U-Net (`TurbNetG`) unchanged. Everything after `TurbNetD` is ours: `VMamba_Block`, `MultiHeadSelfAttention2d`, the configurable `TurbNet`, `MODEL_CONFIGS` (all the tested models) and `build_model()` |
| `dataset.py` | original, small changes | Data loading and normalization (optional fixed train/val split seed) |
| `utils.py` | original, unchanged | Learning-rate schedule, image output |
| `runTrain.py` | original, modified | Training: model chosen by name, fixed seeds, mixed precision, per-epoch metrics (CSV) and loss plot |
| `runTest.py` | original, modified | Test set evaluation: per-sample and test-set relative errors (CSV), bar plot |
| `runTrainCpu.py`, `runTestCpu.py` | original, modified | CPU versions, used only at the start to check that the pipeline worked |
| `run_all_seeds.py` | ours | Trains and tests one model with seeds 42, 1955, 2003 and writes mean ± std |
| `runTuning.py` | ours | Hyperparameter search for the V-Mamba blocks (one training per configuration) |
| `generate_comparison_BigDataset.py` | ours | Test-set comparison table for several models (superseded by `run_all_seeds.py`, included for completeness) |
| `compare_models.py` | ours | Table of the single-seed runs on the reduced set (Table I and Fig. 3 of the report) |
| `vmamba.py` | VMamba repo | `VSSBlock` and selective-scan code |

The comments in the files that come from the original repository mark what we changed.

## Setup

GPU (all our runs: one NVIDIA L4, Linux, Python 3.11, CUDA 12):

```
pip install -r requirements_vm.txt
```

This installs the `mamba-ssm` CUDA selective-scan kernel and Triton, which also needs a C compiler (`gcc`).
Without them `vmamba.py` falls back to a much slower PyTorch implementation.
`requirements_local.txt` is a CPU-only environment, enough for the baseline and for the CPU scripts.

## Data

We use the data sets of the original repository:

* full set, 53.8k samples + 90 test samples: <https://dataserv.ub.tum.de/s/m1459172/download?path=%2F&files=data_full.tar.gz>
* reduced set, 6.4k samples + test samples: <https://dataserv.ub.tum.de/s/m1470791/download?path=%2F&files=data_6k.tar.gz>

Expected layout:

```
data/train/reg/      regular airfoils (full set)
data/train/shear/    sheared airfoils (full set)
data/train/sup/      superimposed airfoils, not used (prop[2] = 0) but the folder must exist, can be empty
data/train_small/    reduced set, all files in one folder
data/test/           90 test samples
```

The scripts load the whole training set in memory (about 42 GB for the full set).

## Models

Models are selected by name, i.e. a key of `MODEL_CONFIGS` in `DfpNet.py`. Each entry says after which
U-Net stage a block is inserted (`out1` ... `out5` encoder, `dout2` ... `dout6` decoder) and which block.

| Report | Name in `MODEL_CONFIGS` |
|---|---|
| Baseline (TurbNet) | `baseline` |
| MonoMamba (final) | `MonoMamba` |
| HalfMamba-Augmented (final) | `HalfMamba_Augmented` |
| Position sweep, one block per stage (Fig. 3) | `M_out1` ... `M_out5`, `A_out1` ... `A_out5` |
| Table I, #2 / #3 (HalfMamba) | `MM____` / `halfMamba` |
| Table I, #4 / #5 / #6 / #7 | `MMMA__` / `halfMambaPush` / `MMAA__` / `MMM_AA` |
| Table I, #8 (all stages) | `fullMamba` |
| Table I, #9 / #10 (decoder blocks) | `HMdec_d2` / `HMdec_d2_d2b`, `HMdec_d2_d2b_d3` |
| Table I, #12 (block on the raw input) | `MambaMentality` |

Table I, #11 (HalfMamba with the depthwise convolution inside the VSS block) is `halfMamba` with
`ssm_conv=3` instead of the default `ssm_conv=1` in `VMamba_Block`.

The tuned hyperparameters (d_state, ssm_ratio, mlp_ratio) are the defaults in `TurbNet.__init__`:
(16, 4, 4) for MonoMamba and (16, 2, 4), (24, 2, 2), (16, 2, 4), (32, 2, 4) for the blocks of
HalfMamba-Augmented at out1, out2, out2b, dout2.

## Running

All commands from `train/`.

**Final comparison on the full set (Table II of the report).** With the default settings (full set, 400k
iterations = 75 epochs, `BIG = 1`, `RUN_TEST = 1`):

```
python run_all_seeds.py baseline
python run_all_seeds.py MonoMamba
python run_all_seeds.py HalfMamba_Augmented
```

Each call trains and tests the model with seeds 42, 1955 and 2003 and writes
`metrics/<model>/summary_<model>_BigDataset.txt` (validation and test metrics, mean ± std over the seeds).

**Single run.**

```
python runTrain.py <model> <run_name> <seed>
python runTest.py  <model> <run_name> <seed>
```

Outputs go to `metrics/<model>/<model>_<run_name>_s<seed>/`: `metrics_train.csv` (one row per epoch: train/val
loss, relative errors per field, time, peak VRAM), `loss_curve_train.png`, the weights `*_modelG`,
`metrics_test.csv` and a sample prediction image in `results/`.

**Exploration on the reduced set (Fig. 3, Table I).** In `dataset.py` set the default `dataDir` to
`"../data/train_small/"` (marked with `#HERE`), in `runTrain.py` set `prop = None` and `iterations = 90000`
(150 epochs), and in `runTest.py` set `prop = None`. Then e.g. `python runTrain.py M_out1 posSweep 42`.
The run folders get a `_small` suffix; in `run_all_seeds.py` set `BIG = 0`. The run names we used are listed
in `compare_models.py`; once the runs exist, `python compare_models.py` builds Table I and the data of Fig. 3
(validation L1 over the last 10 epochs, median time per epoch, peak GPU memory) in `comparison_table.csv`.

**Hyperparameter tuning.** Set `MODEL_TYPE` and the list `EXPERIMENTS` in `runTuning.py` (one row = d_state,
ssm_ratio, mlp_ratio for each tunable block) and run `python runTuning.py`. Every configuration is trained once
(seed 42) and summarized in `metrics/Tuning_<TUNING_LAYER>/summary_<MODEL_TYPE>_<TUNING_LAYER>.csv`, sorted by
the validation error over the last 20 epochs. Our tuning was done on the reduced set.

## Metrics

The relative error of a field c is sum|prediction - target| / sum|target| on the normalized fields, for
c = p, vx, vy and all channels together (aggregate error); accuracy = 100% - aggregate error.

* Validation (`runTrain.py`, every epoch): sums over the 400 validation samples.
* Test (`runTest.py`, `ALL` row of `metrics_test.csv`): sums over the 90 test samples. These are the values
  in the report. On a fixed set this error is proportional to the L1 loss, so both give the same ranking.
  The log file of `runTest.py` also prints the per-sample average of the original code (mean over the
  samples of the per-sample relative error), which gives somewhat different numbers.
* Time per epoch in the report: median of `epoch_time_s` in `metrics_train.csv`; peak GPU memory: mean of
  `peak_vram_mb` (first epoch excluded) / 1000. Both are printed by `run_all_seeds.py` and `compare_models.py`.
  Runs use mixed precision and cuDNN autotuning, so a re-run gives close but not bit-identical numbers.

Expected test results on the full set (relative errors in %, mean ± std over 3 seeds):

| Model | Params | p | vx | vy | Aggregate |
|---|---|---|---|---|---|
| baseline | 2.33 M | 14.51 ± 1.38 | 1.81 ± 0.17 | 6.34 ± 0.60 | 3.06 ± 0.29 |
| MonoMamba | 2.38 M | 12.99 ± 0.54 | 1.62 ± 0.10 | 5.67 ± 0.57 | 2.74 ± 0.19 |
| HalfMamba_Augmented | 2.57 M | 12.48 ± 0.53 | 1.55 ± 0.06 | 5.23 ± 0.14 | 2.60 ± 0.09 |
