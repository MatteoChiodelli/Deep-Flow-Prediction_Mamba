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


MODEL_TYPE   = "HalfMamba_Augmented"      # "HalfMamba" (9 params: 3x3) or "HalfMamba_Augmented" (12 params: 3x4)
TUNING_LAYER = "vmamba_decoder"        # "vmamba1", "vmamba2", "vmamba3", or "vmamba_decoder"
SEED         = 42
EXPO         = 5              

BASELINE_VAL_ERR_PCT = 1.9530 # Average of last 20 epoches of baseline

EXPERIMENTS = [

    [16, 2, 4,   16, 2, 4,   16, 2, 4,   12, 2, 3],
    [16, 2, 4,   16, 2, 4,   16, 2, 4,   16, 1, 2],
    [16, 2, 4,   16, 2, 4,   16, 2, 4,   24 ,2, 2],


]


LAYER_INDEX_MAP = {
    "vmamba1": 0,
    "vmamba2": 3,
    "vmamba3": 6,
    "vmamba_decoder": 9,
}

def fmt_num(x):
    """Formatta int/float senza '.0' superflui."""
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

def get_row_name(model_type, tuning_layer, params, seen_counts):
    idx = LAYER_INDEX_MAP.get(tuning_layer, 0)
    if idx >= len(params):
        raise ValueError(f"TUNING_LAYER='{tuning_layer}' richiede 12 parametri, ma ne hai passati {len(params)}.")
    d_s, s_r, m_r = fmt_num(params[idx]), fmt_num(params[idx+1]), fmt_num(params[idx+2])
    base_name = f"{model_type}_{d_s}_{s_r}_{m_r}"
    
    # Se provi due run che hanno gli stessi 3 valori sul layer target ma cambiano altro, evita sovrascritture
    seen_counts[base_name] = seen_counts.get(base_name, 0) + 1
    if seen_counts[base_name] > 1:
        return f"{base_name}_run{seen_counts[base_name]}"
    return base_name

def summarize_run_csv(csv_path, row_name, params_str, baseline_err_pct):
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
        "val_error_last20_pct": f"{val_mean20_pct:.4f}%",
        "peak_val_error_pct":   f"{val_peak_min_pct:.4f}%",
        "train_error_last20_pct": f"{train_mean20_pct:.4f}%",
        "mean_epoch_time_s":    f"{mean_time_s:.2f}s",
        "mean_vram_mb":         f"{mean_vram_mb:.1f} MB",
        "params":               params_str,
        "improvement_vs_baseline_pct": f"{improvement_pct:+.2f}%",
        "_sort_key": val_mean20_pct
    }


def main():
    tuning_folder_name = f"Tuning_{TUNING_LAYER}"
    tuning_base_dir = os.path.join("metrics", tuning_folder_name)
    os.makedirs(tuning_base_dir, exist_ok=True)
    
    summary_csv_path = os.path.join(tuning_base_dir, f"summary_{MODEL_TYPE}_{TUNING_LAYER}.csv")
    
    summary_rows = []
    seen_counts = {}

    print("=" * 80)
    print(f"AVVIO TUNING AUTOMATICO | Modello: {MODEL_TYPE} | Layer Target: {TUNING_LAYER} | Seed: {SEED}")
    print(f"Cartella di output: {tuning_base_dir}")
    print(f"Totale combinazioni da eseguire: {len(EXPERIMENTS)}")
    print("=" * 80)

    for idx, params in enumerate(EXPERIMENTS, 1):
        mamba_cfg = params_to_mamba_cfg(params)
        row_name = get_row_name(MODEL_TYPE, TUNING_LAYER, params, seen_counts)
        params_str = count_params_millions(MODEL_TYPE, EXPO, mamba_cfg)
        
        print(f"\n>>> [{idx}/{len(EXPERIMENTS)}] Avvio Run: {row_name} ({params_str})")
        print(f"    Configurazione completa: {params}")

        cmd = [
            sys.executable, "runTrain.py",
            MODEL_TYPE,
            row_name,
            str(SEED),
            tuning_folder_name,
        ] + [fmt_num(p) for p in params]

        result = subprocess.run(cmd)
        
        run_metrics_csv = os.path.join(tuning_base_dir, row_name, "metrics_train.csv")
        if result.returncode != 0 or not os.path.exists(run_metrics_csv):
            print(f"[ERRORE] La run {row_name} è fallita o non ha generato {run_metrics_csv}. Salto al prossimo...")
            continue

        row_summary = summarize_run_csv(run_metrics_csv, row_name, params_str, BASELINE_VAL_ERR_PCT)
        summary_rows.append(row_summary)

        fieldnames = [
            "model",
            "val_error_last20_pct",
            "peak_val_error_pct",
            "train_error_last20_pct",
            "mean_epoch_time_s",
            "mean_vram_mb",
            "params",
            "improvement_vs_baseline_pct"
        ]
        with open(summary_csv_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in summary_rows:
                writer.writerow({k: r[k] for k in fieldnames})

        print(f"<<< [{idx}/{len(EXPERIMENTS)}] Completata {row_name}: Val(20)={row_summary['val_error_last20_pct']} | Peak={row_summary['peak_val_error_pct']} | Impr={row_summary['improvement_vs_baseline_pct']}")

    print("\n" + "=" * 80)
    print(f"TUNING COMPLETATO! Riepilogo salvato in: {summary_csv_path}")
    print("=" * 80)
    if summary_rows:
        df_out = pd.DataFrame(summary_rows).drop(columns=["_sort_key"])
        print(df_out.to_string(index=False))

if __name__ == "__main__":
    main()