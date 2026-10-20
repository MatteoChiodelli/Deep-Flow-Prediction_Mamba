import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Importa i moduli dal progetto originale
from dataset import TurbDataset
from DfpNet import build_model

def generate_comparison_table(models_to_test, run_name="BigDataset", seeds=[42, 1955, 2003], data_dir="../data/test/"):
    # Configurazioni di default del progetto
    expo = 5
    prop = [53830, 0.4964, 0.0, 0.5036]
    
    print("Inizializzazione DataLoader di test...")
    test_dataset = TurbDataset(prop, mode=TurbDataset.TEST, dataDirTest=data_dir)
    testLoader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    criterionL1 = nn.L1Loss().cuda()
    
    metrics_base_dir = "metrics"
    os.makedirs(metrics_base_dir, exist_ok=True)
    out_csv_path = os.path.join(metrics_base_dir, "comparison_test_bigdataset.csv")
    
    results = []

    for model_name in models_to_test:
        print(f"\nExperiment: {model_name}_{run_name}")
        print(f"Seeds: {seeds}\n")
        
        # 1. Istanzia il modello per estrarre il conteggio esatto dei parametri
        try:
            netG = build_model(model_name, channelExponent=expo)
            trainable_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
            print(f"Trainable parameters: {trainable_params / 1e6:.2f} Millions ({trainable_params:,} total)\n")
        except Exception as e:
            print(f"Errore nella costruzione del modello {model_name}: {e}")
            continue
            
        seed_metrics = {
            "loss": [], "acc": [], "err_p": [], "err_vx": [], "err_vy": []
        }
        
        # 2. Loop sui seed per il calcolo delle metriche
        for seed in seeds:
            # Cerca il file dei pesi supportando sia la gerarchia annidata che quella piatta
            dir_1 = os.path.join(metrics_base_dir, model_name, f"{model_name}_{run_name}_s{seed}")
            dir_2 = os.path.join(metrics_base_dir, f"{model_name}_{run_name}_s{seed}")
            
            weight_file = None
            for d in [dir_1, dir_2]:
                possible_path = os.path.join(d, f"{model_name}_{run_name}_s{seed}_modelG")
                if os.path.exists(possible_path):
                    weight_file = possible_path
                    break
            
            if not weight_file:
                print(f"  [!] Pesi per il seed {seed} non trovati in {dir_2}. Salto...")
                continue
            
            # Caricamento pesi
            netG.load_state_dict(torch.load(weight_file))
            netG.cuda()
            netG.eval()
            
            L1_accum = 0.0
            val_abs_diff_sum = 0.0; val_target_sum = 0.0
            val_abs_diff_p = 0.0; val_target_p = 0.0
            val_abs_diff_vx = 0.0; val_target_vx = 0.0
            val_abs_diff_vy = 0.0; val_target_vy = 0.0
            
            with torch.no_grad():
                for inputs_cpu, targets_cpu in testLoader:
                    inputs = inputs_cpu.float().cuda()
                    targets = targets_cpu.float().cuda()
                    
                    outputs = netG(inputs)
                    lossL1 = criterionL1(outputs, targets)
                    L1_accum += lossL1.item() * inputs.size(0)
                    
                    # Accumulo errori assoluti e target per il calcolo percentuale
                    val_abs_diff_sum += torch.sum(torch.abs(outputs - targets)).item()
                    val_target_sum += torch.sum(torch.abs(targets)).item()

                    # Canale 0: Pressione
                    val_abs_diff_p += torch.sum(torch.abs(outputs[:, 0, :, :] - targets[:, 0, :, :])).item()
                    val_target_p += torch.sum(torch.abs(targets[:, 0, :, :])).item()
                    
                    # Canale 1: Velocità X
                    val_abs_diff_vx += torch.sum(torch.abs(outputs[:, 1, :, :] - targets[:, 1, :, :])).item()
                    val_target_vx += torch.sum(torch.abs(targets[:, 1, :, :])).item()
                    
                    # Canale 2: Velocità Y
                    val_abs_diff_vy += torch.sum(torch.abs(outputs[:, 2, :, :] - targets[:, 2, :, :])).item()
                    val_target_vy += torch.sum(torch.abs(targets[:, 2, :, :])).item()
            
            # Calcolo metriche relative al termine del dataloader per il seed corrente
            avg_loss = L1_accum / len(test_dataset)
            err_agg = (val_abs_diff_sum / max(val_target_sum, 1e-8))
            acc_agg = max(0.0, (1.0 - err_agg) * 100.0)
            
            err_p_pct = (val_abs_diff_p / max(val_target_p, 1e-8)) * 100.0
            err_vx_pct = (val_abs_diff_vx / max(val_target_vx, 1e-8)) * 100.0
            err_vy_pct = (val_abs_diff_vy / max(val_target_vy, 1e-8)) * 100.0
            
            seed_metrics["loss"].append(avg_loss)
            seed_metrics["acc"].append(acc_agg)
            seed_metrics["err_p"].append(err_p_pct)
            seed_metrics["err_vx"].append(err_vx_pct)
            seed_metrics["err_vy"].append(err_vy_pct)

        if not seed_metrics["loss"]:
            print("Nessun dato processato per questo modello.\n")
            continue

        # 3. Calcolo Media e Deviazione Standard (Varianza sui seed)
        stats = {}
        for k, v in seed_metrics.items():
            stats[f"{k}_mean"] = np.mean(v)
            stats[f"{k}_std"] = np.std(v)

        print("--- TEST EVALUATION (Across Seeds) ---")
        print(f"Test Loss final:            {stats['loss_mean']:.6f} +/- {stats['loss_std']:.6f}")
        print(f"Test Accuracy final:        {stats['acc_mean']:.2f}% +/- {stats['acc_std']:.2f}%\n")
        print(f"Test Error Pressure (p):    {stats['err_p_mean']:.2f}% +/- {stats['err_p_std']:.2f}%")
        print(f"Test Error Velocity X (vx): {stats['err_vx_mean']:.2f}% +/- {stats['err_vx_std']:.2f}%")
        print(f"Test Error Velocity Y (vy): {stats['err_vy_mean']:.2f}% +/- {stats['err_vy_std']:.2f}%\n")

        # Preparazione riga per il CSV
        results.append({
            "Model": model_name,
            "Run Name": run_name,
            "Trainable Params": trainable_params,
            "Test Loss": f"{stats['loss_mean']:.6f} +/- {stats['loss_std']:.6f}",
            "Test Accuracy (%)": f"{stats['acc_mean']:.2f} +/- {stats['acc_std']:.2f}",
            "Err P (%)": f"{stats['err_p_mean']:.2f} +/- {stats['err_p_std']:.2f}",
            "Err Vx (%)": f"{stats['err_vx_mean']:.2f} +/- {stats['err_vx_std']:.2f}",
            "Err Vy (%)": f"{stats['err_vy_mean']:.2f} +/- {stats['err_vy_std']:.2f}"
        })

    # 4. Scrittura del file CSV riassuntivo
    if results:
        df = pd.DataFrame(results)
        # Se il file esiste, appendiamo le righe, altrimenti lo creiamo nuovo
        mode = 'a' if os.path.exists(out_csv_path) else 'w'
        header = not os.path.exists(out_csv_path)
        df.to_csv(out_csv_path, mode=mode, header=header, index=False)
        print(f"Tabella comparativa aggiornata con successo in: {out_csv_path}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Genera tabella comparativa sui test set per più modelli.")
    
    # Accept one or more model names (nargs='+')
    parser.add_argument("models", nargs="+", help="Nomi dei modelli (es. HalfMamba_Augmented Baseline MambaMentality)")
    parser.add_argument("--run_name", type=str, default="BigDataset", help="Nome della run (default: BigDataset)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1955, 2003], help="Lista di seed da valutare")
    parser.add_argument("--data_dir", type=str, default="../data/test/", help="Path della cartella test data")

    args = parser.parse_args()

    # Esegue la funzione passandogli la lista di modelli ricevuta da CLI
    generate_comparison_table(
        models_to_test=args.models, 
        run_name=args.run_name, 
        seeds=args.seeds,
        data_dir=args.data_dir
    )