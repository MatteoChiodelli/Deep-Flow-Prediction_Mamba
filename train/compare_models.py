import pandas as pd
import os

SAVE = 1
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(CURRENT_DIR, "metrics")

# Formato: (model, run, seed, custom_path)
# Inserisci None o ometti il quarto elemento se vuoi il calcolo automatico del path.
MODELS_TO_PROCESS = [
    ("baseline", "exp1", "s42", None),
    ("halfMamba", "lightconv1", "s42", None),
    ("HalfMamba", "run2", "seed2", "Tuning_all_layers_HalfMamba/Combination_1/metrics_train.csv")
]

def process_metrics():
    records = []
    
    for item in MODELS_TO_PROCESS:
        model = item[0]
        
        # Se c'è un quarto elemento valido, usa il custom path e solo il nome del modello
        if len(item) == 4 and item[3] is not None:
            path = f"{BASE_DIR}/{item[3]}"
            model_run_name = model
        else:
            # Altrimenti calcola il path in automatico e usa model_run
            run = item[1]
            seed = item[2]
            path = f"{BASE_DIR}/{model}/{model}_{run}_{seed}_small/metrics_train.csv"
            model_run_name = f"{model}_{run}"
        
        try:
            df = pd.read_csv(path)
            
            cols_last_20 = ['val_accuracy_pct', 'val_rel_error', 'train_accuracy_pct', 'val_err_p', 'val_err_vx', 'val_err_vy']
            mean_last_20 = df[cols_last_20].tail(20).mean()
            
            cols_except_first = ['epoch_time_s', 'peak_vram_mb']
            mean_except_first = df[cols_except_first].iloc[1:].mean()
            
            min_val_rel_err = df['val_rel_err'].min() if 'val_rel_err' in df.columns else df['val_rel_error'].min()
            
            row = {
                'Model': model_run_name,
                'val_accuracy_pct_last20': mean_last_20['val_accuracy_pct'],
                'val_rel_error_last20': mean_last_20['val_rel_error'],
                'train_accuracy_pct_last20': mean_last_20['train_accuracy_pct'],
                'val_err_p_last20': mean_last_20['val_err_p'],
                'val_err_vx_last20': mean_last_20['val_err_vx'],
                'val_err_vy_last20': mean_last_20['val_err_vy'],
                'epoch_time_s_avg_excl_first': mean_except_first['epoch_time_s'],
                'peak_vram_mb_avg_excl_first': mean_except_first['peak_vram_mb'],
                'val_rel_err_min': min_val_rel_err
            }
            records.append(row)
            
        except Exception as e:
            print(f"Error processing {path}: {e}")

    if not records:
        print("No valid data processed.")
        return

    result_df = pd.DataFrame(records).set_index('Model')

    if SAVE == 1:
        result_df.to_csv("comparison_table.csv")
        print("Results successfully saved to comparison_table.csv")
    else:
        print(result_df.to_string())

if __name__ == "__main__":
    process_metrics()