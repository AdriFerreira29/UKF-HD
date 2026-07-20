"""Agrega results_paper.csv por configuração, calculando MAE/RMSE média±desvio sobre os trials.

Uso: python scripts/aggregate_paper.py [results_paper.csv]
Gera results_paper_summary.csv e imprime a tabela.
"""
import sys
import pandas as pd

infile = sys.argv[1] if len(sys.argv) > 1 else "results_paper.csv"
df = pd.read_csv(infile)

group_cols = ["Dataset", "Model", "encoder", "online", "use_backprop", "ukf", "ukf_mode",
              "models", "Gaussian", "Poisson", "MissingP"]
df["MAE"] = pd.to_numeric(df["MAE"], errors="coerce")
df["RMSE"] = pd.to_numeric(df["RMSE"], errors="coerce")

agg = (df.groupby(group_cols, dropna=False)
         .agg(MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
              RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
              n=("MAE", "count"))
         .reset_index())

agg.to_csv("results_paper_summary.csv", index=False)
print(agg.to_string(index=False))
print("\nResumo salvo em results_paper_summary.csv")
