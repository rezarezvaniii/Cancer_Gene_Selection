"""
Diagnostic-only probe (not part of the pipeline): does widening the Phase-1 SHAP
pool from 60 to 150 let RIME-LF close the gap to the paper on Colon/CNS?
Uses MEDIUM-equivalent search settings, only for these two small datasets. Writes to
a separate CSV so it never touches outputs/summary.csv (main.py's checkpoint).

Run from the project root:
    python probe_pool_size.py
"""
import pandas as pd

from src.data_loader import load_and_clean_data
from src.shapley_filter import get_kSV_features
from src.model_selector import run_pipeline

DATASETS = ["Colon.arff", "CNS.arff"]
POOL_SIZE = 150

rows = []
for filename in DATASETS:
    print(f"\n=== {filename} (pool={POOL_SIZE}) ===")
    X, y, classes = load_and_clean_data(f"data/{filename}")
    top_genes, _ = get_kSV_features(X, y, top_n=POOL_SIZE)
    X_reduced = X[top_genes]

    result = run_pipeline(
        X_reduced, y, dataset_name=filename,
        n_runs=1,
        n_agents=80,
        max_iter=100,
        penalty_coef=0.001,
        final_eval_repeats=15,
        n_jobs=-1,
        patience=20,
        search_cv_splits=5,
        search_cv_repeats=2,
        n_top_candidates=3,
        algorithm="RIME-LF",
    )
    best = result["best"]
    print(f"RESULT {filename}: avg={best['average']*100:.2f}% high={best['highest']*100:.2f}% "
          f"low={best['lowest']*100:.2f}% n_genes={best['n_genes']}")
    rows.append({
        "Dataset": filename,
        "Pool size": POOL_SIZE,
        "Average": f"{best['average']*100:.2f}%",
        "Highest": f"{best['highest']*100:.2f}%",
        "Lowest": f"{best['lowest']*100:.2f}%",
        "Selected Genes": best["n_genes"],
    })

df = pd.DataFrame(rows)
print("\n" + df.to_string(index=False))
df.to_csv("outputs/pool_size_probe_150.csv", index=False)
print("\nSaved to outputs/pool_size_probe_150.csv")
