"""
ساخت دیتاست Leukemia-3c از همون golub.csv که برای Leukemia-4c استفاده کردی
(دانلودشده از https://www.openintro.org/data/index.php?data=golub)

برخلاف Leukemia-4c (که باید AML رو بر اساس BM.PB می‌شکستی)، اینجا ساده‌تره:
ستون "cancer" از قبل دقیقاً همون ۳ کلاس رو داره (allB, allT, aml) — فقط ستون‌های
غیرژنی رو حذف می‌کنیم و "cancer" رو به‌عنوان کلاس نهایی نگه می‌داریم.

استفاده:
    python make_leukemia_3c.py golub.csv Leukemia_3c.csv
"""
import sys
import pandas as pd

def main(in_path, out_path):
    df = pd.read_csv(in_path)

    if "cancer" not in df.columns:
        print("❌ ستون 'cancer' توی فایل پیدا نشد — مطمئن شو همون golub.csv درسته.")
        print(f"   ستون‌های موجود (اول و آخر): {list(df.columns[:5])} ... {list(df.columns[-5:])}")
        sys.exit(1)

    print("توزیع کلاس‌ها:")
    print(df["cancer"].value_counts())

    df_final = df.rename(columns={"cancer": "class"})
    drop_cols = [c for c in ["Samples", "BM.PB", "Gender", "Source", "tissue.mf"] if c in df_final.columns]
    df_final = df_final.drop(columns=drop_cols)

    df_final.to_csv(out_path, index=False)
    print(f"\n✅ ذخیره شد: {out_path}  (شکل: {df_final.shape[0]} نمونه × {df_final.shape[1]-1} ژن + ۱ ستون کلاس)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("استفاده: python make_leukemia_3c.py golub.csv Leukemia_3c.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
