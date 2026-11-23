"""
Ablation Study — اثر هر جزء از SHAP-RIME-LF رو جداگانه اندازه می‌گیره.

چهار حالت تست می‌شه:
  1) Full (SHAP-RIME-LF)      — فیلتر SHAP + RIME-LF + کانفیگ خودکار SVM (روش کامل)
  2) w/o SHAP                 — فیلتر SHAP با انتخاب تصادفی ژن جایگزین می‌شه (بقیه ثابت)
  3) w/o Lévy Flight          — RIME-LF با RIME خام (بدون Lévy Flight) جایگزین می‌شه (بقیه ثابت)
  4) w/o Auto-Config          — کانفیگ خودکار SVM با یه تنظیم ثابت (rbf, C=1.0) جایگزین می‌شه (بقیه ثابت)

⚠️ به‌خاطر محدودیت زمانی، این اسکریپت فقط روی یه زیرمجموعه‌ی نماینده از دیتاست‌ها
اجرا می‌شه (نه هر ۱۴ تا) — طبق DATASETS_SUBSET پایین. اگه وقت داری همه‌ی ۱۴ تا رو
اضافه کن، ولی همین ۴-۵ تا هم برای نشون‌دادن روند (کدوم جزء مهم‌تره) کافیه.

اجرا:
    python run_ablation.py

خروجی: outputs/ablation_results.csv (میانگین روی دیتاست‌های تست‌شده، برای هر ۴ حالت)
"""

import os
import numpy as np
import pandas as pd

from src.data_loader import load_and_clean_data
from src.shapley_filter import get_kSV_features
from src.optimizers.common import auto_select_kernel, evaluate_final_subset
from src.optimizers.rime_lf_optimizer import RIMELF_Optimizer
from src.optimizers.rime_only_optimizer import RIMEOnly_Optimizer

# ===================== تنظیمات =====================
# ⚠️ اگه وقت داری، بقیه‌ی دیتاست‌ها رو هم اضافه کن؛ این‌ها فقط یه زیرمجموعه‌ی
# نماینده‌ان (ترکیبی از دیتاست کوچیک/بزرگ، و اونایی که قبلاً باهاشون کار کردیم)
DATASETS_SUBSET = ["Colon.arff", "CNS.arff", "Leukemia-3c.csv", "Lung_Cancer.xlsx"]

TOP_N_SHAPLEY = 60
N_AGENTS = 60
MAX_ITER = 80
PATIENCE = 15
N_RUNS = 2
PENALTY_COEF = 0.001
SEARCH_CV_SPLITS = 5
SEARCH_CV_REPEATS = 2
N_TOP_CANDIDATES = 3
FIXED_SVM_CFG = {"kernel": "rbf", "C": 1.0, "gamma": "scale", "class_weight": None}


def run_variant(X_reduced, y, dataset_name, svm_cfg, optimizer_class, n_runs=N_RUNS):
    """یک run_pipeline ساده‌شده، برای استفاده‌ی مجدد بین حالت‌های ablation."""
    run_averages = []
    for run_idx in range(n_runs):
        seed = 1000 * (run_idx + 1)
        optimizer = optimizer_class(
            X_reduced, y, svm_cfg=svm_cfg, n_wolves=N_AGENTS, max_iter=MAX_ITER,
            penalty_coef=PENALTY_COEF, n_jobs=-1, verbose=False, seed=seed,
            patience=PATIENCE, search_cv_splits=SEARCH_CV_SPLITS,
            search_cv_repeats=SEARCH_CV_REPEATS, n_top_candidates=N_TOP_CANDIDATES,
        )
        genes, fitness, history, top_candidates = optimizer.optimize()
        best_genes, _ = top_candidates[0]
        metrics = evaluate_final_subset(X_reduced, y, best_genes, svm_cfg, n_repeats=10)
        run_averages.append(metrics["average"])
    return float(np.mean(run_averages)), float(np.mean([len(best_genes)]))


def random_filter(X, y, top_n, seed=42):
    """جایگزین فیلتر SHAP برای حالت w/o SHAP — انتخاب کاملاً تصادفی ژن."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[1], size=min(top_n, X.shape[1]), replace=False)
    return X.columns[idx].tolist()


def main():
    os.makedirs("outputs", exist_ok=True)
    rows = []

    for filename in DATASETS_SUBSET:
        filepath = f"data/{filename}"
        if not os.path.exists(filepath):
            print(f"❌ '{filepath}' پیدا نشد — رد شد.")
            continue

        print(f"\n{'='*60}\n📁 {filename}\n{'='*60}")
        X, y, _ = load_and_clean_data(filepath)

        # فیلتر SHAP معمولی (برای Full، w/o Lévy Flight، و w/o Auto-Config)
        shap_genes, _ = get_kSV_features(X, y, top_n=TOP_N_SHAPLEY)
        X_shap = X[shap_genes]

        # فیلتر تصادفی (فقط برای w/o SHAP)
        rand_genes = random_filter(X, y, TOP_N_SHAPLEY)
        X_rand = X[rand_genes]

        # کانفیگ خودکار SVM (برای Full، w/o SHAP، w/o Lévy Flight)
        auto_cfg = auto_select_kernel(X_shap, y, verbose=False)

        print("▶️  Full (SHAP + RIME-LF + Auto-Config) ...")
        acc_full, ngenes_full = run_variant(X_shap, y, filename, auto_cfg, RIMELF_Optimizer)
        print(f"   acc={acc_full*100:.2f}% genes={ngenes_full:.0f}")

        print("▶️  w/o SHAP (random filter + RIME-LF + Auto-Config) ...")
        acc_noshap, ngenes_noshap = run_variant(X_rand, y, filename, auto_cfg, RIMELF_Optimizer)
        print(f"   acc={acc_noshap*100:.2f}% genes={ngenes_noshap:.0f}")

        print("▶️  w/o Lévy Flight (SHAP + RIME-only + Auto-Config) ...")
        acc_norime, ngenes_norime = run_variant(X_shap, y, filename, auto_cfg, RIMEOnly_Optimizer)
        print(f"   acc={acc_norime*100:.2f}% genes={ngenes_norime:.0f}")

        print("▶️  w/o Auto-Config (SHAP + RIME-LF + fixed SVM) ...")
        acc_nocfg, ngenes_nocfg = run_variant(X_shap, y, filename, FIXED_SVM_CFG, RIMELF_Optimizer)
        print(f"   acc={acc_nocfg*100:.2f}% genes={ngenes_nocfg:.0f}")

        rows.append({"Dataset": filename, "Variant": "Full (SHAP-RIME-LF)", "Accuracy": acc_full*100, "Genes": ngenes_full})
        rows.append({"Dataset": filename, "Variant": "w/o SHAP filter", "Accuracy": acc_noshap*100, "Genes": ngenes_noshap})
        rows.append({"Dataset": filename, "Variant": "w/o Lévy Flight", "Accuracy": acc_norime*100, "Genes": ngenes_norime})
        rows.append({"Dataset": filename, "Variant": "w/o Auto-Config", "Accuracy": acc_nocfg*100, "Genes": ngenes_nocfg})

        pd.DataFrame(rows).to_csv("outputs/ablation_results.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n" + "📊"*15)
    print("   خلاصه: میانگین روی همه‌ی دیتاست‌های تست‌شده")
    print("📊"*15)
    summary = df.groupby("Variant")[["Accuracy", "Genes"]].mean().round(2)
    summary = summary.reindex(["Full (SHAP-RIME-LF)", "w/o SHAP filter", "w/o Lévy Flight", "w/o Auto-Config"])
    print(summary)
    summary.to_csv("outputs/ablation_summary.csv")
    print("\n💾 ذخیره شد: outputs/ablation_results.csv و outputs/ablation_summary.csv")


if __name__ == "__main__":
    main()
