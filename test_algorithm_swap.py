"""
تست سریع جایگزینی GWO در IGWO با یک الگوریتم دیگه (طبق دستور استاد):
فقط روی ۲ دیتاست کوچیک/سریع (Colon, CNS — ضعیف‌ترین میانگین‌ها در outputs/summary.csv فعلی،
پس سخت‌ترین محک) با تنظیمات سبک اجرا می‌کنه و نتیجه رو کنار baseline فعلی IGWO می‌ذاره.
اگه نتیجه نزدیک بود (افت محسوس نداشت) می‌شه رفت سراغ اجرای کامل main.py؛ اگه نه، برو سراغ
الگوریتم بعدی توی src/model_selector.py -> ALGORITHMS.

استفاده:
    python test_algorithm_swap.py RIME-LF
    python test_algorithm_swap.py IGO
    python test_algorithm_swap.py ISBOA
"""

import sys
import os
import pandas as pd

from src.data_loader import load_and_clean_data
from src.shapley_filter import get_kSV_features
from src.model_selector import run_pipeline, ALGORITHMS

DATASETS_TO_TEST = ["Colon.arff", "CNS.arff"]

# باید دقیقاً از outputs/summary.csv فعلی (IGWO) خونده بشه — اگه اون فایل عوض شد این‌ها رو هم آپدیت کن
BASELINE_AVG = {
    "Colon.arff": 88.33,
    "CNS.arff": 95.89,
}

TOP_N_SHAPLEY = 60

# تنظیمات سبک برای تست سریع (نه تنظیمات کامل main.py)
N_AGENTS = 60
MAX_ITER = 50
PATIENCE = 15
N_RUNS = 1
FINAL_EVAL_REPEATS = 10
SEARCH_CV_SPLITS = 4
SEARCH_CV_REPEATS = 1
N_TOP_CANDIDATES = 3
PENALTY_COEF = 0.001

DROP_THRESHOLD_PP = 2.0  # حداکثر افت مجاز (واحد درصد) نسبت به baseline تا "قابل قبول" باشه


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ALGORITHMS:
        print(f"❌ استفاده: python test_algorithm_swap.py <ALGORITHM>   (یکی از {list(ALGORITHMS)})")
        sys.exit(1)
    algorithm = sys.argv[1]

    rows = []
    for filename in DATASETS_TO_TEST:
        filepath = f"data/{filename}"
        if not os.path.exists(filepath):
            print(f"❌ '{filepath}' پیدا نشد — رد شد.")
            continue

        print("\n" + "=" * 60)
        print(f"🚀 {algorithm} روی {filename}")
        print("=" * 60)

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
            patience=PATIENCE,
            search_cv_splits=SEARCH_CV_SPLITS,
            search_cv_repeats=SEARCH_CV_REPEATS,
            n_top_candidates=N_TOP_CANDIDATES,
            algorithm=algorithm,
        )

        best = result["best"]
        avg_pct = best["average"] * 100
        baseline = BASELINE_AVG.get(filename)
        drop = baseline - avg_pct if baseline is not None else None
        verdict = "✅ قابل‌قبول" if (drop is not None and drop <= DROP_THRESHOLD_PP) else "❌ افت محسوس"

        rows.append({
            "Dataset": filename,
            "Algorithm": algorithm,
            f"{algorithm} Avg": f"{avg_pct:.2f}%",
            "IGWO Baseline Avg": f"{baseline:.2f}%" if baseline is not None else "N/A",
            "Delta (pp)": f"{-drop:+.2f}" if drop is not None else "N/A",
            "Genes": best["n_genes"],
            "Verdict": verdict,
        })

    if rows:
        df = pd.DataFrame(rows)
        print("\n" + "📊" * 15)
        print(f"   نتیجه‌ی تست سریع: {sys.argv[1]}")
        print("📊" * 15)
        print(df.to_string(index=False))
        os.makedirs("outputs", exist_ok=True)
        out_path = f"outputs/algorithm_swap_{algorithm}.csv"
        df.to_csv(out_path, index=False)
        print(f"\n💾 ذخیره شد در {out_path}")

        if all("✅" in r["Verdict"] for r in rows):
            print(f"\n✅ {algorithm} روی هر دو دیتاست قابل‌قبوله — می‌تونی اجرای کامل main.py رو با این الگوریتم بزنی.")
        else:
            print(f"\n❌ {algorithm} روی حداقل یک دیتاست افت محسوس داشت — برو سراغ الگوریتم بعدی.")


if __name__ == "__main__":
    main()
