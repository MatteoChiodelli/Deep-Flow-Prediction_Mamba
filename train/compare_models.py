################
#
# Validation comparison table (our script)
#
# Reads the metrics_train.csv of single-seed runs on the reduced dataset and builds one row per run.
# The first columns are the ones of the report (Table I and Fig. 3): validation and training L1 loss
# averaged over the last 10 epochs, median time per epoch and peak GPU memory. The other columns are
# relative errors averaged over the last 20 epochs (the metric used by runTuning.py) and the best
# validation error. Output: comparison_table.csv (SAVE = 1) or printed table.
#
################

import pandas as pd
import os
import glob

SAVE = 1
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(CURRENT_DIR, "metrics")

# Format: (label, model, run, seed, custom_path)
# The run folder is the one written by runTrain.py: metrics/<model>/<model>_<run>_<seed>_small/.
# custom_path (relative to metrics/) is only tried if that folder has no metrics_train*.csv with data.
MODELS_TO_PROCESS = [
    # Table I
    ("#0 baseline",                 "baseline",        "exp1",       "s42", None),
    ("#1 M: out1",                  "M_out1",          "posSweep",   "s42", None),
    ("#2 M: out1, out2",            "MM____",          "exp1",       "s42", None),
    ("#3 HalfMamba",                "halfMamba",       "lightconv1", "s42", None),
    ("#4 #3 + A: out3",             "MMMA__",          "exp1",       "s42", None),
    ("#5 M: out1, out2; A: out2b",  "halfMambaPush",   "Light",      "s42", None),
    ("#6 M: out1, out2; A: out2b, out3", "MMAA__",     "exp1",       "s42", None),
    ("#7 #3 + A: out4, out5",       "MMM_AA",          "exp1",       "s42", None),
    ("#8 M: all stages",            "fullMamba",       "exp1",       "s42", "metrics_train_fullMamba.csv"),
    ("#9 HalfMamba-Aug.",           "HMdec_d2",        "decSweep",   "s42", None),
    ("#10 #9 + M: dout2b",          "HMdec_d2_d2b",    "decSweep",   "s42", None),
    ("#10 #9 + M: dout2b, dout3",   "HMdec_d2_d2b_d3", "decSweep",   "s42", None),
    ("#11 #3 with inner DWConv",    "halfMamba",       "exp1",       "s42", None),  # trained with ssm_conv=3 in VMamba_Block
    ("#12 #3 + M on raw input",     "MambaMentality",  "convo",      "s42", None),
    # Fig. 3 (position sweep), one block after a single encoder stage
] + [(f"Fig3 M: {s}", f"M_{s}", "posSweep", "s42", None) for s in ["out1", "out2", "out2b", "out3", "out4", "out5"]] \
  + [(f"Fig3 A: {s}", f"A_{s}", "posSweep", "s42", None) for s in ["out1", "out2", "out2b", "out3", "out4", "out5"]]

def load_run(model, run, seed, custom_path):
    # standard file first, then any metrics_train*.csv in the run folder (some of ours were renamed),
    # then the custom path; returns the first one with at least one epoch
    folder = f"{BASE_DIR}/{model}/{model}_{run}_{seed}_small"
    candidates = [f"{folder}/metrics_train.csv"] + sorted(glob.glob(f"{folder}/metrics_train*.csv"))
    if custom_path is not None:
        candidates.append(f"{BASE_DIR}/{custom_path}")
    for path in candidates:
        if os.path.isfile(path):
            df = pd.read_csv(path, skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]
            df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["val_loss"])
            if len(df) > 0:
                return df
    raise FileNotFoundError(f"no metrics_train csv with data in {folder}")

def process_metrics():
    records = []

    for label, model, run, seed, custom_path in MODELS_TO_PROCESS:
        try:
            df = load_run(model, run, seed, custom_path)

            cols_last_20 = ['val_accuracy_pct', 'val_rel_error', 'train_accuracy_pct', 'val_err_p', 'val_err_vx', 'val_err_vy']
            mean_last_20 = df[cols_last_20].tail(20).mean()

            cols_except_first = ['epoch_time_s', 'peak_vram_mb']
            mean_except_first = df[cols_except_first].iloc[1:].mean()

            row = {
                'Model': label,
                'run': f"{model}_{run}_{seed}",
                # as in the report: L1 x 1e-3 over the last 10 epochs, median s/epoch, peak memory in GB (MB/1000)
                'val_L1_last10_e-3': df['val_loss'].tail(10).mean() * 1e3,
                'train_L1_last10_e-3': df['train_loss'].tail(10).mean() * 1e3,
                'epoch_time_s_median': df['epoch_time_s'].median(),
                'peak_vram_gb': mean_except_first['peak_vram_mb'] / 1000,
                'val_accuracy_pct_last20': mean_last_20['val_accuracy_pct'],
                'val_rel_error_last20': mean_last_20['val_rel_error'],
                'train_accuracy_pct_last20': mean_last_20['train_accuracy_pct'],
                'val_err_p_last20': mean_last_20['val_err_p'],
                'val_err_vx_last20': mean_last_20['val_err_vx'],
                'val_err_vy_last20': mean_last_20['val_err_vy'],
                'epoch_time_s_avg_excl_first': mean_except_first['epoch_time_s'],
                'peak_vram_mb_avg_excl_first': mean_except_first['peak_vram_mb'],
                'val_rel_err_min': df['val_rel_error'].min()
            }
            records.append(row)

        except Exception as e:
            print(f"Error processing {label} ({model}_{run}_{seed}): {e}")

    if not records:
        print("No valid data processed.")
        return

    result_df = pd.DataFrame(records).set_index('Model')

    print(result_df[['val_L1_last10_e-3', 'train_L1_last10_e-3', 'epoch_time_s_median', 'peak_vram_gb']].round(2).to_string())
    if SAVE == 1:
        result_df.to_csv("comparison_table.csv")
        print("Results successfully saved to comparison_table.csv")

if __name__ == "__main__":
    process_metrics()
