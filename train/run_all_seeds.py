import subprocess
import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MODEL = "mambaSup"        # any name in DfpNet.MODEL_CONFIGS: 'baseline', 'mambaSup', 'fullMamba', 'C0'...'C4'
RUN_NAME = "exp1"
SEEDS = [42, 1955, 2003]
BIG = 0 # 1= Big Dataset 
        # Remember to also update the files runTrain.py,runTest.py,dataset.py with the big dataset (search #HERE)
SAVE = 1        # 1 = Save
RUN_TEST = 0    # 1 = Test
GPU = 1 # Using gpu
        # Recommended: only use Big=1 when GPU=1
# ------------------------------------

if GPU:
    TRAIN_SCRIPT_PATH = "runTrain.py"
    TEST_SCRIPT_PATH  = "runTest.py"
else:
    TRAIN_SCRIPT_PATH = "runTrainCpu.py"  
    TEST_SCRIPT_PATH  = "runTestCpu.py"

if BIG:
    suffix_str = ""
else:
    suffix_str = "_small"

print(f"=== Starting multi seed experiment for: {MODEL} ===")

trainable_parameters = None
for seed in SEEDS:
    print(f"\n--- Launching TRAINING {MODEL} with Seed: {seed} ---")
    cmd = [
        "python",
        "-u",
        TRAIN_SCRIPT_PATH,
        MODEL,
        RUN_NAME,
        str(seed)
    ]

    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out_lines = []
    
    if proc.stdout:
        for line in proc.stdout:
            print(line, end="", flush=True)
            out_lines.append(line)
    proc.wait()

    class ResultObj:
        stdout: str = ""
        returncode: int = 0
        
    result = ResultObj()
    result.stdout = "".join(out_lines)
    result.returncode = proc.returncode

    parameter_match = re.search(
        r"Initialized .* with ([\d,]+) trainable params",
        result.stdout,
    )
    if parameter_match and trainable_parameters is None:
        trainable_parameters = int(parameter_match.group(1).replace(",", ""))

    if result.returncode != 0:
        print(f"Error during training seed: {seed}!")
        break

if RUN_TEST:
    for seed in SEEDS:
        print(f"\n--- Launching TESTING {MODEL} with Seed: {seed} ---")
        cmd = [
            "python",
            "-u",
            TEST_SCRIPT_PATH,
            MODEL,
            RUN_NAME,
            str(seed)
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.stdout:
            for line in proc.stdout:
                print(line, end="", flush=True) 
        proc.wait()

print("\n=== Computing Final Metrics ===")
final_val_losses = []
final_val_accs = []

final_val_err_p = []
final_val_err_vx = []
final_val_err_vy = []

all_train_losses = []
all_val_losses = []

final_test_err_p = []
final_test_err_v = []
final_test_err_agg = []
final_test_accs = []

for seed in SEEDS:
    folder_name = f"{MODEL}_{RUN_NAME}_s{seed}{suffix_str}"
    metrics_dir = os.path.join("metrics", MODEL, folder_name)
    
    train_csv = os.path.join(metrics_dir, "metrics_train.csv")
        
    if os.path.isfile(train_csv):
        df = pd.read_csv(train_csv)
        last_row = df.iloc[-1]
        final_val_losses.append(last_row["val_loss"])
        final_val_accs.append(last_row["val_accuracy_pct"])

        if "val_err_p" in df.columns:
            final_val_err_p.append(last_row["val_err_p"])
            final_val_err_vx.append(last_row["val_err_vx"])
            final_val_err_vy.append(last_row["val_err_vy"])
        
        all_train_losses.append(df["train_loss"].values)
        all_val_losses.append(df["val_loss"].values)

    if RUN_TEST:
        test_csv = os.path.join(metrics_dir, "metrics_test.csv")
        if os.path.isfile(test_csv):
            df_test = pd.read_csv(test_csv)
            row = df_test[df_test['sample_id'].astype(str) == "ALL"]
            if not row.empty:
                final_test_err_p.append(row.iloc[-1]["rel_error_pressure_pct"])
                final_test_err_v.append(row.iloc[-1]["rel_error_velocity_pct"])
                final_test_err_agg.append(row.iloc[-1]["rel_error_aggregate_pct"])
                final_test_accs.append(row.iloc[-1]["accuracy_aggregate_pct"])

if len(final_val_losses) == len(SEEDS):
    mean_loss = np.mean(final_val_losses)
    std_loss  = np.std(final_val_losses)
    mean_acc  = np.mean(final_val_accs)
    std_acc   = np.std(final_val_accs)

    if len(final_val_err_p) == len(SEEDS):
        mean_vp = np.mean(final_val_err_p)
        std_vp = np.std(final_val_err_p)
        mean_vvx = np.mean(final_val_err_vx)
        std_vvx = np.std(final_val_err_vx)
        mean_vvy = np.mean(final_val_err_vy)
        std_vvy = np.std(final_val_err_vy)
    
if RUN_TEST and len(final_test_accs) == len(SEEDS):
    mean_err_p   = np.mean(final_test_err_p)
    std_err_p    = np.std(final_test_err_p)
    mean_err_v   = np.mean(final_test_err_v)
    std_err_v    = np.std(final_test_err_v)
    mean_err_agg = np.mean(final_test_err_agg)
    std_err_agg  = np.std(final_test_err_agg)
    mean_tacc    = np.mean(final_test_accs)
    std_tacc     = np.std(final_test_accs)
    
summary_lines = []
summary_lines.append(f"Experiment: {MODEL}_{RUN_NAME}")
summary_lines.append(f"Seeds: {SEEDS}\n")

if trainable_parameters is not None:
    summary_lines.append(    
        "Trainable parameters: "
        f"{trainable_parameters / 1_000_000:.2f} Millions " 
        f"({trainable_parameters:,} total)\n"
    )

if len(final_val_losses) == len(SEEDS):
    summary_lines.append("--- VALIDATION (During Training) ---")
    summary_lines.append(f"Validation Loss final:       {mean_loss:.6f} +/- {std_loss:.6f}")
    summary_lines.append(f"Validation Accuracy final:   {mean_acc:.2f}% +/- {std_acc:.2f}%\n")
    if len(final_val_err_p) == len(SEEDS):
        summary_lines.append(f"Val Error Pressure (p):      {mean_vp:.2f}% +/- {std_vp:.2f}%")
        summary_lines.append(f"Val Error Velocity X (vx):   {mean_vvx:.2f}% +/- {std_vvx:.2f}%")
        summary_lines.append(f"Val Error Velocity Y (vy):   {mean_vvy:.2f}% +/- {std_vvy:.2f}%\n")
    else:
        summary_lines.append("")
    
if RUN_TEST and len(final_test_accs) == len(SEEDS):
    summary_lines.append("--- TESTING (Unseen Data) ---")
    summary_lines.append(f"Rel Error Pressure (%):      {mean_err_p:.2f}% +/- {std_err_p:.2f}%")
    summary_lines.append(f"Rel Error Velocity (%):      {mean_err_v:.2f}% +/- {std_err_v:.2f}%")
    summary_lines.append(f"Rel Error Aggregate (%):     {mean_err_agg:.2f}% +/- {std_err_agg:.2f}%")
    summary_lines.append(f"Test Accuracy final:         {mean_tacc:.2f}% +/- {std_tacc:.2f}%\n")

summary_text = "\n".join(summary_lines)

print("\n" + "="*45)
print(summary_text)
print("="*45)

if SAVE:
    model_parent_dir = os.path.join("metrics", MODEL)
    os.makedirs(model_parent_dir, exist_ok=True)
    summary_file = os.path.join(model_parent_dir, f"summary_{MODEL}_{RUN_NAME}{suffix_str}.txt")

    with open(summary_file, "w") as f:
        f.write(summary_text)
    print(f"\nRecup saved in {summary_file}")
else:
    print("\n[INFO] SAVE=0: Nessun file di riepilogo testuale salvato su disco. Preparazione del grafico a schermo...")
    
    if all_train_losses and all_val_losses:
        plt.figure(figsize=(10, 6))
        
        min_len = min([len(t) for t in all_train_losses])
        avg_train = np.mean([t[:min_len] for t in all_train_losses], axis=0)
        avg_val = np.mean([v[:min_len] for v in all_val_losses], axis=0)
        
        plt.plot(range(1, min_len + 1), avg_train, label="Average Train Loss (L1)", color="#1f77b4", linewidth=2)
        plt.plot(range(1, min_len + 1), avg_val, label="Average Validation Loss (L1)", color="#ff7f0e", linestyle="--", linewidth=2)
        
        for i in range(len(all_train_losses)):
            plt.plot(range(1, len(all_train_losses[i]) + 1), all_train_losses[i], color="#1f77b4", alpha=0.15)
            plt.plot(range(1, len(all_val_losses[i]) + 1), all_val_losses[i], color="#ff7f0e", alpha=0.15, linestyle="--")

        plt.title(f"{MODEL.capitalize()} - Aggregated Loss Over {len(SEEDS)} Seeds", fontsize=14)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("L1 Loss", fontsize=12)
        
        if len(final_val_losses) > 0:
            final_txt = f"Final Val Loss: {mean_loss:.5f} $\\pm$ {std_loss:.5f}\nFinal Val Acc: {mean_acc:.2f}% $\\pm$ {std_acc:.2f}%"
            plt.annotate(final_txt, xy=(0.55, 0.65), xycoords='axes fraction', 
                        fontsize=11, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.9))
            
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=11, loc="upper right")
        plt.tight_layout()
        
        plt.show()

if RUN_TEST:
    print("\n Cleaning Temporary Test Logs...")
    for seed in SEEDS:
        testout_file = f"{MODEL}_{RUN_NAME}_s{seed}_testout.txt"
        if os.path.exists(testout_file):
            os.remove(testout_file)
    print("Done.")
