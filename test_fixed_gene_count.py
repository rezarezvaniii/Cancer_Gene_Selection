"""
تمرین استاد: «دیتاست‌هایی که مقاله ازت بهتره رو با همون تعداد ژنی که مقاله گزارش
کرده تست کن، ببین می‌تونی به همون دقت (یا نزدیک بهش) برسی یا نه.»

این اسکریپت جدا از main.py هست چون هدفش فرق داره: اینجا IGWO رو مجبور نمی‌کنیم
خودش به یک تعداد ژن خاص برسه (که کار سختیه و به penalty_coef حساسه)، بلکه مستقیم
دقیقاً K تا از بهترین ژن‌های فاز ۱ (رتبه‌بندی kSV/SHAP) رو می‌گیریم و ارزیابی می‌کنیم.
اگه این عدد هم پایین بود، مشکل «تعداد ژن» نیست، «کیفیت رتبه‌بندی ژن‌ها»ست.

نحوه‌ی استفاده: مقادیر PAPER_GENE_COUNTS رو (طبق جدول ۱۲ مقاله) عوض نکن مگر بخوای
عدد دیگه‌ای تست کنی. فقط دیتاست‌هایی که می‌خوای تست بشن رو در DATASETS_TO_TEST بذار.
"""

import os
import pandas as pd

from src.data_loader import load_and_clean_data
from src.shapley_filter import get_kSV_features
from src.optimizers.common import auto_select_kernel, evaluate_topk_subset

# طبق جدول ۱۲ مقاله (ستون kSV-IGWO): چند تا ژن مقاله برای هر دیتاست گزارش کرده
# (این عدد رو صرفاً برای مقایسه استفاده می‌کنیم -- خود مقاله تعداد دقیق ژن هر سطر رو
# در جداول ۴ تا ۱۱ آورده؛ این‌ها بهترین ردیف هر جدول‌اند، یعنی جایی که بالاترین دقت
# average ثبت شده)
PAPER_GENE_COUNTS = {
    "Colon.arff": 21,          # جدول ۴: 21 ژن -> average 98.51%
    "CNS.arff": 12,            # جدول ۱۰: 12 ژن -> average 99.25%
    "MLL.arff": 8,             # جدول ۹: 8 ژن -> average 100%
    "SRBCT.arff": 18,          # جدول ۶: 18 ژن -> average 100%
    "Prostate_Tumor.arff": 8,  # جدول ۱۱: 8 ژن -> average 98.77%
    "Lung.arff": 12,           # جدول ۵: 12 ژن -> average 97.99%
}

# فقط دیتاست‌هایی که استادت گفته "مقاله بهترن" رو اینجا بذار (Lymphoma و DLBCL رو که
# خودت بهتری، نیازی به تست دوباره نیست)
DATASETS_TO_TEST = ["Colon.arff", "CNS.arff", "MLL.arff", "SRBCT.arff"]

TOP_N_SHAPLEY_POOL = 60   # فاز ۱ همیشه یک استخر ۶۰ تایی می‌سازه؛ از داخل همین استخر K تا برمی‌داریم
FINAL_EVAL_REPEATS = 30   # دقیقاً مثل مقاله (Section 3: تکرار ۳۰ باره)


def main():
    rows = []
    for filename in DATASETS_TO_TEST:
        filepath = f"data/{filename}"
        if not os.path.exists(filepath):
            print(f"❌ '{filepath}' پیدا نشد — رد شد.")
            continue

        k = PAPER_GENE_COUNTS.get(filename)
        if k is None:
            print(f"⚠️  عدد ژن مقاله برای {filename} تعریف نشده — رد شد.")
            continue

        print("\n" + "=" * 60)
        print(f"🚀 {filename}  (K = {k} ژن، طبق مقاله)")
        print("=" * 60)

        X, y, classes = load_and_clean_data(filepath)
        ranked_genes, importance_df = get_kSV_features(X, y, top_n=max(TOP_N_SHAPLEY_POOL, k))
        svm_cfg = auto_select_kernel(X[ranked_genes], y, verbose=True)

        result = evaluate_topk_subset(
            X, y, ranked_genes, k=k, svm_cfg=svm_cfg, n_repeats=FINAL_EVAL_REPEATS
        )

        print(f"\n📌 نتیجه با دقیقاً {result['k']} ژن (بالاترین‌های SHAP):")
        print(f"   Highest={result['highest']*100:.2f}%  "
              f"Average={result['average']*100:.2f}%  "
              f"Lowest={result['lowest']*100:.2f}%")

        rows.append({
            "Dataset": filename,
            "Genes (paper K)": k,
            "Highest": f"{result['highest']*100:.2f}%",
            "Average": f"{result['average']*100:.2f}%",
            "Lowest": f"{result['lowest']*100:.2f}%",
        })

    if rows:
        print("\n" + "📊" * 15)
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        os.makedirs("outputs", exist_ok=True)
        df.to_csv("outputs/fixed_gene_count_comparison.csv", index=False)
        print("\n💾 ذخیره شد در outputs/fixed_gene_count_comparison.csv")
        print("   این جدول رو مستقیم می‌تونی کنار جدول مقاله بذاری و مقایسه کنی.")


if __name__ == "__main__":
    main()
