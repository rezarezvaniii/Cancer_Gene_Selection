"""
اجرای pipeline کامل بر اساس مقاله kSV-IGWO (Afreen et al., Knowl Inf Syst 2025):
  Phase 1: فیلتر kSV (Kernel Shapley Value, تقریب Tree-SHAP) -> top-N ژن
  Phase 2: IGWO (GWO کلاسیک + استراتژی DLH طبق Eq. 8-11 مقاله)
  Phase 3: ارزیابی نهایی به سبک مقاله (Highest / Average / Lowest روی چند تکرار CV)

توجه مهم نسبت به نسخه‌ی قبلی:
  - HHO حذف شد. مقاله فقط kSV-IGWO رو با kSV تنها و GWO تنها مقایسه می‌کنه،
    HHO اصلاً بخشی از مقاله نیست.
  - فقط یک کانفیگ (طبق جدول ۳ مقاله) اجرا می‌شه، نه ده‌ها حالت مختلف.
  - لیست دیتاست‌ها با ۸ دیتاست مقاله (جدول ۲) یکی شد.

برای اجرا:
    pip install -r requirements.txt
    python main.py
"""

import os
import json
import pandas as pd

from src.data_loader import load_and_clean_data
from src.shapley_filter import get_kSV_features
from src.model_selector import run_pipeline
from src.visualizer import plot_dataset_results, plot_summary_bar

# ============================== CONFIG (طبق جدول ۳ مقاله) ==============================
# اسم فایل‌ها رو با چیزی که واقعاً توی پوشه‌ی data/ داری تطبیق بده (.arff یا .csv یا .xlsx/.xls)
DATASET_FILES = [
    "Lung_Cancer.xlsx",
    "DLBCL.xlsx",
    # "Leukemia.arff",
    # "MLL.arff",
    # "Lung.arff",
    # "Lymphoma.arff",
    # "DLBCL.arff",
    # "CNS.arff",
    # "Prostate_Tumor.arff",
    # "Colon.arff",
    # "SRBCT.arff",
]

TOP_N_SHAPLEY = 60          # Table 3: Selected features (F) = 60  — یکسان برای همه‌ی دیتاست‌ها
N_AGENTS = 150               # Table 3: Search agent number = 150
MAX_ITER = 200                # Table 3: Maximum iterations = 200 (سقف بالا؛ عملاً با PATIENCE زودتر متوقف می‌شه)
PATIENCE = 30                 # اگه ۳۰ ایتریشن پشت سر هم بهبود نبود، IGWO زودتر متوقف می‌شه (early stopping)
N_RUNS = 1                   # قبلاً ۳ بود؛ چون خیلی زمان‌بره کم شد. اگه بعداً خواستی پایداری نتیجه رو
                              # بیشتر بسنجی می‌تونی به ۲ یا ۳ برش گردونی، ولی هر واحد اضافه یعنی زمان کل ×N
FINAL_EVAL_REPEATS = 10     # تکرار RepeatedStratifiedKFold برای ارزیابی نهایی (مقاله از ۳۰ استفاده کرده،
                              # اینجا برای زمان اجرا کمتره؛ اگه وقت داری بذارش 30 تا دقیقاً مثل مقاله بشه)
PENALTY_COEF = 0.001         # جریمه‌ی خیلی کوچک برای انتخاب ژن کمتر (طبق متن مقاله)
N_JOBS = -1
OUTPUT_DIR = "outputs"
# ==========================================================================================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_summary = []
    detailed_results = {}

    for filename in DATASET_FILES:
        print("\n" + "=" * 60)
        print(f"🚀 PROCESSING DATASET: {filename}")
        print("=" * 60)

        filepath = f"data/{filename}"
        if not os.path.exists(filepath):
            print(f"❌ Error: File '{filepath}' not found! Skipping...")
            continue

        try:
            X, y, classes = load_and_clean_data(filepath)

            top_genes, _ = get_kSV_features(X, y, top_n=TOP_N_SHAPLEY)
            X_reduced = X[top_genes]

            result = run_pipeline(
                X_reduced, y, dataset_name=filename,
                n_runs=N_RUNS,
                n_agents=N_AGENTS,
                max_iter=MAX_ITER,
                penalty_coef=PENALTY_COEF,
                final_eval_repeats=FINAL_EVAL_REPEATS,
                n_jobs=N_JOBS,
                patience=PATIENCE,
            )

            best = result["best"]
            final_summary.append({
                "Dataset": filename,
                "Best Avg Accuracy": f"{best['average']*100:.2f}%",
                "Highest": f"{best['highest']*100:.2f}%",
                "Lowest": f"{best['lowest']*100:.2f}%",
                "Selected Genes": best["n_genes"],
                "SVM Kernel": result["svm_cfg"]["kernel"],
            })
            detailed_results[filename] = {
                "best_genes": best["genes"],
                "average_accuracy": best["average"],
                "svm_cfg": result["svm_cfg"],
            }

            plot_dataset_results(filename, result["all_results"], out_dir=OUTPUT_DIR)

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "📊" * 15)
    print("   FINAL RESULTS (kSV-IGWO)")
    print("📊" * 15)
    if final_summary:
        summary_df = pd.DataFrame(final_summary)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
        with open(os.path.join(OUTPUT_DIR, "selected_genes.json"), "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        plot_summary_bar(final_summary, out_dir=OUTPUT_DIR)
        print(f"\n💾 Results saved to '{OUTPUT_DIR}/' (summary.csv, selected_genes.json, plots)")
    else:
        print("No results to display.")
    print("=" * 40)


if __name__ == "__main__":
    main()