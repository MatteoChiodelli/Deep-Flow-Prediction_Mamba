################
#
# Automatic Hyperparameter Tuning Runner for DfpNet (Mamba blocks)
#
################

import os
import sys
import csv
import subprocess
import pandas as pd
from DfpNet import build_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_TYPE   = "HalfMamba_Augmented"      # "HalfMamba" (9 params: 3x3) or "HalfMamba_Augmented" (12 params: 3x4)
TUNING_LAYER = "all_layers"        # "vmamba1", "vmamba2", "vmamba3", or "vmamba_decoder"
SEED         = 42
EXPO         = 5              

BASELINE_VAL_ERR_PCT = 1.9530 # Average of last 20 epoches of baseline

EXPERIMENTS = [ 


    [16, 2, 2,   16, 2, 2,   16, 2, 2,   32, 2, 4],
    [24, 2, 2,   24, 2, 2,   24, 2, 2,   32, 2, 4],
    [12, 2, 3,   12, 2, 3,   12, 2, 3,   32, 2, 4],
    [16, 1, 2,   16, 1, 2,   16, 1, 2,   32, 2, 4],
    [32, 2, 4,   32, 2, 4,   32, 2, 4,   32, 2, 4],
    [16, 2, 2,   24, 2, 2,   24, 2, 2,   32, 2, 4],
    [16, 2, 2,   32, 2, 4,   32, 2, 4,   32, 2, 4],
    [12, 2, 2,   24, 2, 3,   32, 2, 4,   32, 2, 4],
    [32, 2, 4,   24, 2, 3,   12, 2, 2,   32, 2, 4],


]


LAYER_INDEX_MAP = {
    "vmamba1": 0,
    "vmamba2": 3,
    "vmamba3": 6,
    "vmamba_decoder": 9,
    "all_layers":0,
}

def fmt_num(x):
    return str(int(x)) if float(x).is_integer() else str(x)

def params_to_mamba_cfg(params):
    if len(params) not in (9, 12):
        raise ValueError(f"Ogni esperimento deve avere 9 o 12 parametri, trovati {len(params)}: {params}")
    cfg = {
        "vmamba1": {"ssm_d_state": params[0], "ssm_ratio": params[1], "mlp_ratio": params[2]},
        "vmamba2": {"ssm_d_state": params[3], "ssm_ratio": params[4], "mlp_ratio": params[5]},
        "vmamba3": {"ssm_d_state": params[6], "ssm_ratio": params[7], "mlp_ratio": params[8]},
    }
    if len(params) == 12:
        cfg["vmamba_decoder"] = {"ssm_d_state": params[9], "ssm_ratio": params[10], "mlp_ratio": params[11]}
    return cfg

def count_params_millions(model_type, expo, mamba_cfg):
    net = build_model(model_type, channelExponent=expo, dropout=0.0, mamba_cfg=mamba_cfg)
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    del net
    return f"{total_params / 1e6:.2f} Mln"

def params_to_str(params):
    # e.g. "16-2-4_16-2-4_16-2-4_12-2-3": one group (d_state-ssm_ratio-mlp_ratio) per layer
    return "_".join("-".join(fmt_num(p) for p in params[i:i+3]) for i in range(0, len(params), 3))

def get_row_name(model_type, tuning_layer, params, seen_counts):
    if tuning_layer == "all_layers":
        base_name = f"{model_type}_{params_to_str(params)}"
    else:
        idx = LAYER_INDEX_MAP[tuning_layer]
        if idx >= len(params):
            raise ValueError(f"TUNING_LAYER='{tuning_layer}' richiede 12 parametri, ma ne hai passati {len(params)}.")
        d_s, s_r, m_r = fmt_num(params[idx]), fmt_num(params[idx+1]), fmt_num(params[idx+2])
        base_name = f"{model_type}_{d_s}_{s_r}_{m_r}"
    
    # Se provi due run che hanno gli stessi 3 valori sul layer target ma cambiano altro, evita sovrascritture
    seen_counts[base_name] = seen_counts.get(base_name, 0) + 1
    if seen_counts[base_name] > 1:
        return f"{base_name}_run{seen_counts[base_name]}"
    return base_name

def summarize_run_csv(csv_path, row_name, config_str, params_str, baseline_err_pct):
    df = pd.read_csv(csv_path)
    last20 = df.tail(20)
    
    val_mean20_pct   = last20["val_rel_error"].mean() * 100.0
    val_peak_min_pct = df["val_rel_error"].min() * 100.0
    train_mean20_pct = last20["train_rel_error"].mean() * 100.0
    
    df_steady = df[df["epoch"] >= 2] if len(df) > 1 else df
    mean_time_s  = df_steady["epoch_time_s"].mean()
    mean_vram_mb = df_steady["peak_vram_mb"].mean()
    
    improvement_pct = ((baseline_err_pct - val_mean20_pct) / baseline_err_pct) * 100.0
    
    return {
        "model": row_name,
        "config": config_str,
        "val_error_last20_pct": f"{val_mean20_pct:.4f}%",
        "peak_val_error_pct":   f"{val_peak_min_pct:.4f}%",
        "train_error_last20_pct": f"{train_mean20_pct:.4f}%",
        "mean_epoch_time_s":    f"{mean_time_s:.2f}s",
        "mean_vram_mb":         f"{mean_vram_mb:.1f} MB",
        "params":               params_str,
        "improvement_vs_baseline_pct": f"{improvement_pct:+.2f}%",
        "_sort_key": val_mean20_pct
    }


FIELDNAMES = [
    "model",
    "config",
    "val_error_last20_pct",
    "peak_val_error_pct",
    "train_error_last20_pct",
    "mean_epoch_time_s",
    "mean_vram_mb",
    "params",
    "improvement_vs_baseline_pct",
]

def load_previous_summary(summary_csv_path):
    # keeps the rows of earlier tuning sessions, so re-running with a new EXPERIMENTS list does not lose them
    if not os.path.exists(summary_csv_path):
        return []
    df = pd.read_csv(summary_csv_path, dtype=str).fillna("")
    rows = df.to_dict("records")
    for r in rows:
        r["_sort_key"] = float(str(r.get("val_error_last20_pct", "inf")).rstrip("%") or "inf")
    return rows

def write_summary(summary_csv_path, summary_rows):
    summary_rows.sort(key=lambda r: r["_sort_key"])
    with open(summary_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, restval="", extrasaction="ignore")
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

def main():
    # runTrain.py, ../data and metrics/ are all relative to the train/ folder
    os.chdir(SCRIPT_DIR)
    tuning_folder_name = f"Tuning_{TUNING_LAYER}"
    tuning_base_dir = os.path.join("metrics", tuning_folder_name)
    os.makedirs(tuning_base_dir, exist_ok=True)
    
    summary_csv_path = os.path.join(tuning_base_dir, f"summary_{MODEL_TYPE}_{TUNING_LAYER}.csv")
    
    summary_rows = load_previous_summary(summary_csv_path)
    seen_counts = {}

    print("=" * 80)
    print(f"ACITVATING AUTO TRAIN | Modello: {MODEL_TYPE} | Layer Target: {TUNING_LAYER} | Seed: {SEED}")
    print(f"Output Folder: {tuning_base_dir}")
    print(f"Total number of combinations: {len(EXPERIMENTS)}")
    print("=" * 80)
    for idx, params in enumerate(EXPERIMENTS, 1):
        mamba_cfg = params_to_mamba_cfg(params)
        row_name = get_row_name(MODEL_TYPE, TUNING_LAYER, params, seen_counts)
        config_str = params_to_str(params)
        params_str = count_params_millions(MODEL_TYPE, EXPO, mamba_cfg)
        print(f"\n>>> [{idx}/{len(EXPERIMENTS)}] Starting Run: {row_name} ({params_str})")
        print(f"    Configuration Done: {params}")

        cmd = [
            sys.executable, "runTrain.py",
            MODEL_TYPE,
            row_name,
            str(SEED),
            tuning_folder_name,
        ] + [fmt_num(p) for p in params]

        result = subprocess.run(cmd, env={**os.environ, "DFP_EXPO": str(EXPO)})
        
        run_metrics_csv = os.path.join(tuning_base_dir, row_name, "metrics_train.csv")
        if result.returncode != 0 or not os.path.exists(run_metrics_csv):
            print(f"[ERROR] Run {row_name} failerd o did not generate {run_metrics_csv}. Going next...")
            continue

        row_summary = summarize_run_csv(run_metrics_csv, row_name, config_str, params_str, BASELINE_VAL_ERR_PCT)
        # a re-run of the same config replaces its old row
        summary_rows = [r for r in summary_rows if r["model"] != row_name]
        summary_rows.append(row_summary)
        write_summary(summary_csv_path, summary_rows)

        print(f"<<< [{idx}/{len(EXPERIMENTS)}] Completated {row_name}: Val(20)={row_summary['val_error_last20_pct']} | Peak={row_summary['peak_val_error_pct']} | Impr={row_summary['improvement_vs_baseline_pct']}")

    print("\n" + "=" * 80)
    print(f"TUNING COMPLETE! Riepilogo salvato in: {summary_csv_path}")
    print("=" * 80)
    if summary_rows:
        df_out = pd.DataFrame(summary_rows)[FIELDNAMES]
        print(df_out.to_string(index=False))

if __name__ == "__main__":
    main()
