
import os
import sys
import json
import pandas as pd

# وقتی خروجی به فایل/Tee-Object پایپ می‌شه، ویندوز cp1252 می‌ده و پرینت فارسی کرش می‌کنه
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from src.data_loader import load_and_clean_data
from src.shapley_filter import get_kSV_features
from src.model_selector import run_pipeline
from src.visualizer import plot_dataset_results, plot_summary_bar


DATASET_FILES = [
    # ---- ۱۱ دیتاست میکروآرایه ----                      # نمونه / ژن / کلاس
    # "Colon.arff",                  # Colon Tumor          62 /  2000 / 2   ✔️ تموم شده
    # "CNS.arff",                    # CNS                  60 /  7129 / 2   ✔️ تموم شده
    # "Ovarian.arff",                # Ovarian             253 / 15154 / 2   ✔️ تموم شده
    # "Leukemia.arff",               # ALL-AML              72 /  7129 / 2   ✔️ تموم شده
    # "Breast.arff",                 # Breast Cancer        97 / 24481 / 2   ✔️ تموم شده
    # "Prostate_Tumor.csv",          # Prostate Tumor      102 / 10509 / 2   ✔️ تموم شده
    # "MLL.arff",                    # MLL                  72 / 12582 / 3   ✔️ تموم شده
    # "Leukemia-3c.csv",             # Leukemia-3c          72 /  7129 / 3   ✔️ تموم شده
    # "Leukemia_4c.csv",             # Leukemia-4c          72 /  7129 / 4   ✔️ تموم شده
    "SRBCT.arff",                    # SRBCT                83 /  2308 / 4   🔁 اجرای دوباره (قبلی: outputs_30runs/backup_SRBCT_run1)
    # "Lung_Cancer.xlsx",            # Lung Cancer         203 / 12600 / 5   ✔️ تموم شده

    # ---- ۳ دیتاست Omic (CuMiDa) ----
    # "Breast_GSE10797.arff",        # Breast (Omic)        66 / 22278 / 3   ✔️ تموم شده
    # "Lung_GSE18842.arff",          # Lung (Omic)          90 / 54676 / 2   ✔️ تموم شده
    # "Leukemia_GSE22529_U133A.arff",# Leukemia (Omic)      52 / 22284 / 2   ✔️ تموم شده
]

TOP_N_SHAPLEY = 60          # Table 3: Selected features (F) = 60  — یکسان برای همه‌ی دیتاست‌ها
PENALTY_COEF = 0.001         # جریمه‌ی خیلی کوچک برای انتخاب ژن کمتر (طبق متن مقاله)
N_JOBS = -1
OUTPUT_DIR = "outputs_30runs"   # پوشه‌ی جدا تا با نتایج ۲-اجرایی قبلی قاطی نشه

# طبق دستور استاد: DLH (که از مقاله‌ی Afreen et al. کپی شده بود) حذف شد. OBL هم امتحان
# شد ولی روی Colon/CNS بهبودی نداشت، پس با Lévy Flight جایگزین شد.
# برای برگشت به IGWO اصلی یا امتحان GO/SBOA فقط همین مقدار رو عوض کن:
# یکی از "IGWO", "RIME-LF", "IGO", "ISBOA" (رجوع کن به src/model_selector.py -> ALGORITHMS)
ALGORITHM = "RIME-LF"

# RUN_MODE سه‌حالته: "FAST" (چند دقیقه، فقط برای تست اینکه کد اجرا می‌شه)،
# "MEDIUM" (چند ساعت — برای گرفتن نتیجه‌ی اولیه‌ی قابل‌دفاع و نشون دادن به استاد،
# قبل از قفل کردن نسخه‌ی نهایی)، "FULL" (طبق جدول ۳ مقاله دقیقاً — چند روز، فقط
# برای عدد نهایی که توی مقاله چاپ می‌شه، بعد از اینکه با MEDIUM مطمئن شدی همه‌چی
# درست کار می‌کنه و مشکل خاصی نیست).
#
# ⚠️ چرا FULL این‌قدر طول می‌کشه: با N_AGENTS=150 و MAX_ITER=200 و N_RUNS=3،
# هر دیتاست می‌تونه صدها هزار بار SVM رو با cross-validation ارزیابی کنه (کش
# میان‌بر می‌زنه ولی خیلی از زیرمجموعه‌ها یکتا هستن). MEDIUM حدود ۵-۸ برابر
# سریع‌تر از FULL است ولی هنوز جستجوی معناداری انجام می‌ده.
# "PAPER": حالت اجرای نهایی مقاله با ۳۰ اجرای مستقل (seed های 1000, 2000, ..., 30000).
# پارامترهای جستجو مثل MEDIUM هستن تا ۳۰ اجرا در زمان معقول تموم بشه؛ اگه وقت دارید
# می‌تونید PAPER_SEARCH_LEVEL رو "FULL" کنید (خیلی کندتر). اجرا run به run چک‌پوینت
# می‌شه، پس اگه قطع شد فقط دوباره python main.py بزنید، از همون run ادامه می‌ده.
RUN_MODE = "PAPER"  # یکی از "FAST", "MEDIUM", "FULL", "PAPER"
PAPER_SEARCH_LEVEL = "MEDIUM"  # "MEDIUM" یا "FULL" — فقط برای RUN_MODE="PAPER"

if RUN_MODE == "PAPER" and PAPER_SEARCH_LEVEL == "MEDIUM":
    MAX_ITER = 100
    PATIENCE = 20
    N_RUNS = 30                   # ۳۰ اجرای مستقل
    FINAL_EVAL_REPEATS = 30
    SEARCH_CV_SPLITS = 5
    SEARCH_CV_REPEATS = 2
    N_TOP_CANDIDATES = 3
    N_AGENTS = 80
elif RUN_MODE == "PAPER":  # PAPER + FULL (Table 3)
    MAX_ITER = 200
    PATIENCE = 40
    N_RUNS = 30
    FINAL_EVAL_REPEATS = 30
    SEARCH_CV_SPLITS = 5
    SEARCH_CV_REPEATS = 3
    N_TOP_CANDIDATES = 3
    N_AGENTS = 150
elif RUN_MODE == "FAST":
    MAX_ITER = 80
    PATIENCE = 15
    N_RUNS = 1
    FINAL_EVAL_REPEATS = 10
    SEARCH_CV_SPLITS = 4
    SEARCH_CV_REPEATS = 2
    N_TOP_CANDIDATES = 5
    N_AGENTS = 60
elif RUN_MODE == "MEDIUM":
    # هدف: چیزی که روی یه لپ‌تاپ معمولی توی چند ساعت (نه چند روز) تموم بشه، ولی هنوز
    # جستجوی جدی انجام بده — برای نتیجه‌ی اولیه که به استاد نشون بدی و مطمئن بشی
    # مسیر درسته، قبل از اینکه FULL رو (که چند روز طول می‌کشه) قفل کنی.
    MAX_ITER = 100
    PATIENCE = 20
    N_RUNS = 2
    FINAL_EVAL_REPEATS = 15
    SEARCH_CV_SPLITS = 5
    SEARCH_CV_REPEATS = 2
    N_TOP_CANDIDATES = 3
    N_AGENTS = 80
else:  # FULL — طبق جدول ۳ مقاله دقیقاً (برای عدد نهایی مقاله)
    MAX_ITER = 200                # Table 3: Maximum iterations = 200 (سقف بالا؛ عملاً با PATIENCE زودتر متوقف می‌شه)
    PATIENCE = 40                 # اگه ۴۰ ایتریشن پشت سر هم بهبود نبود، زودتر متوقف می‌شه
    N_RUNS = 3                    # ۳ اجرای مستقل با seed های متفاوت (شانس بیشتر برای فرار از optimum محلی)
    FINAL_EVAL_REPEATS = 30      # دقیقاً مثل مقاله (Section 3: 30 تکرار)
    SEARCH_CV_SPLITS = 5
    SEARCH_CV_REPEATS = 3        # فیتنس جستجو روی میانگین ۳ تقسیم‌بندی متفاوت (نه یک fold ثابت)
                                  # تا overfitting به یک split خاص کمتر بشه (دیدیم با ۱ split،
                                  # فیتنس جستجو ۰.۹۱۷۹ می‌شد ولی ارزیابی واقعی فقط ۸۵.۸۷٪ بود)
    N_TOP_CANDIDATES = 3         # چند تا از بهترین کاندیدهای متمایز جستجو رو با ارزیابی دقیق
                                  # می‌سنجیم (نه فقط alpha) — کمتر از FAST_MODE چون هزینه‌ی هر
                                  # کاندید با ۳۰ تکرار بیشتره و N_RUNS=3 هم خودش هزینه رو ضرب می‌کنه
    N_AGENTS = 150               # Table 3: Search agent number = 150
# ==========================================================================================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    genes_path = os.path.join(OUTPUT_DIR, "selected_genes.json")

    # ===== Checkpoint/Resume: اگه قبلاً یه بار (کامل یا ناقص) اجرا کردی و اجرا قطع
    # شده (کرش، قطعی برق، بستن ترمینال)، دیتاست‌هایی که قبلاً نتیجه‌شون ذخیره شده
    # دوباره از اول اجرا نمی‌شن. اگه می‌خوای همه رو از صفر اجرا کنی، summary.csv و
    # selected_genes.json رو دستی پاک کن یا RESUME=False کن. =====
    RESUME = True
    final_summary, detailed_results = [], {}
    already_done = set()
    if RESUME and os.path.exists(summary_path) and os.path.exists(genes_path):
        try:
            prev_df = pd.read_csv(summary_path)
            with open(genes_path, "r", encoding="utf-8") as f:
                detailed_results = json.load(f)
            # چک‌پوینت‌های قدیمی (قبل از تغییر اسم ستون‌ها) رو به اسکیمای فعلی نگاشت می‌کنیم
            # تا جدول نهایی همیشه ستون‌های یکسان داشته باشه (بدون NaN پراکنده بین ردیف‌های
            # قدیمی/جدید) و نموداری که از رو همین جدول ساخته می‌شه کرش نکنه. summary.csv فعلی
            # از قبل هر دو ستون قدیمی و جدید رو هم‌زمان داره (از یک اجرای نصفه‌کاره با اسکیمای
            # درحال‌تغییر)، پس به‌جای rename ساده (که باعث دو ستون هم‌نام می‌شد) مقدار خالی
            # ستون جدید رو از ستون قدیمی پر می‌کنیم و بعد ستون قدیمی رو حذف می‌کنیم.
            for old_col, new_col in [
                ("Best Avg Accuracy", "Average Accuracy (mean of n_runs)"),
                ("Highest", "Highest (best run)"),
                ("Lowest", "Lowest (worst run)"),
            ]:
                if old_col in prev_df.columns:
                    if new_col in prev_df.columns:
                        prev_df[new_col] = prev_df[new_col].fillna(prev_df[old_col])
                    else:
                        prev_df[new_col] = prev_df[old_col]
                    prev_df = prev_df.drop(columns=[old_col])
            already_done = set(prev_df["Dataset"]) & set(detailed_results.keys())
            final_summary = prev_df[prev_df["Dataset"].isin(already_done)].to_dict("records")
            if already_done:
                print(f"♻️  Resume: {len(already_done)} دیتاست از اجرای قبلی پیدا شد و دوباره اجرا نمی‌شه: "
                      f"{sorted(already_done)}")
        except Exception as e:
            print(f"⚠️  خوندن چک‌پوینت قبلی شکست خورد ({e}) — از صفر شروع می‌شه.")
            final_summary, detailed_results, already_done = [], {}, set()

    for filename in DATASET_FILES:
        if filename in already_done:
            print(f"⏭️  {filename} از چک‌پوینت قبلی رد شد (قبلاً تموم شده).")
            continue

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
                search_cv_splits=SEARCH_CV_SPLITS,
                search_cv_repeats=SEARCH_CV_REPEATS,
                n_top_candidates=N_TOP_CANDIDATES,
                algorithm=ALGORITHM,
                checkpoint_dir=os.path.join(OUTPUT_DIR, "run_checkpoints"),
            )

            best = result["best"]
            final_summary.append({
                "Dataset": filename,
                "N Runs": best["n_runs"],
                "Average Accuracy (mean of n_runs)": f"{best['average']*100:.2f}%",
                "Std": f"{best['std']*100:.2f}",
                "Highest (best run)": f"{best['highest']*100:.2f}%",
                "Lowest (worst run)": f"{best['lowest']*100:.2f}%",
                "Selected Genes": best["n_genes"],
                "Mean Selected Genes": f"{best['mean_n_genes']:.1f}",
                "SVM Kernel": result["svm_cfg"]["kernel"],
            })

            # همه‌ی ۳۰ عدد هر run جدا ذخیره می‌شه — برای جدول ضمیمه و آزمون Wilcoxon
            per_run_path = os.path.join(OUTPUT_DIR, "per_run_results.csv")
            per_run_df = pd.DataFrame([{
                "Dataset": filename, "Run": r["run"], "Seed": 1000 * r["run"],
                "Accuracy (%)": round(r["average"] * 100, 4),
                "Highest (%)": round(r["highest"] * 100, 4),
                "Lowest (%)": round(r["lowest"] * 100, 4),
                "N Genes": r["n_genes"],
            } for r in result["all_results"]])
            if os.path.exists(per_run_path):
                old = pd.read_csv(per_run_path)
                per_run_df = pd.concat([old[old["Dataset"] != filename], per_run_df], ignore_index=True)
            per_run_df.to_csv(per_run_path, index=False)
            detailed_results[filename] = {
                "best_genes": best["genes"],
                "average_accuracy": best["average"],
                "svm_cfg": result["svm_cfg"],
            }

            plot_dataset_results(filename, result["all_results"], out_dir=OUTPUT_DIR)

            # ===== چک‌پوینت: بعد از هر دیتاست فوراً ذخیره می‌شه، نه فقط آخر کار — تا
            # اگه اجرا وسط راه (مثلاً روی دیتاست ۵ از ۸) قطع شد، نتیجه‌ی ۴ تای قبلی
            # از دست نره و لازم نباشه از اول اجرا کنی. =====
            pd.DataFrame(final_summary).to_csv(summary_path, index=False)
            with open(genes_path, "w", encoding="utf-8") as f:
                json.dump(detailed_results, f, ensure_ascii=False, indent=2)
            print(f"💾 چک‌پوینت ذخیره شد ({len(final_summary)} دیتاست تا الان تموم شده).")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "📊" * 15)
    print(f"   FINAL RESULTS (SHAP-{ALGORITHM})")
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