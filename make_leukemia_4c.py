"""
ساخت دیتاست Leukemia-4c از فایل golub.csv (دانلودشده از
https://www.openintro.org/data/index.php?data=golub)

استفاده:
    python make_leukemia_4c.py golub.csv Leukemia_4c.csv
"""
import sys
import pandas as pd

def build_4c_label(row):
    if row["cancer"] == "aml" and row["BM.PB"] == "BM":
        return "AML_BM"
    elif row["cancer"] == "aml" and row["BM.PB"] == "PB":
        return "AML_PB"
    elif row["cancer"] == "allB":
        return "ALL_B"
    elif row["cancer"] == "allT":
        return "ALL_T"
    else:
        return None  # اگه چیزی مطابقت نداشت (نباید پیش بیاد، ولی برای اطمینان)

def main(in_path, out_path):
    df = pd.read_csv(in_path)
    df["class"] = df.apply(build_4c_label, axis=1)

    missing = df["class"].isna().sum()
    if missing:
        print(f"⚠️ هشدار: {missing} ردیف لیبل نگرفت — چک کن ستون‌های cancer/BM.PB درست خونده شدن.")

    print("توزیع کلاس‌ها:")
    print(df["class"].value_counts())

    # ستون‌های غیرژنی (metadata) رو حذف می‌کنیم، فقط ژن‌ها + کلاس نهایی می‌مونه —
    # دقیقاً فرمتی که main.py/data_loader.py انتظار داره (ستون‌های ژن + یک ستون کلاس)
    drop_cols = [c for c in ["Samples", "BM.PB", "Gender", "Source", "tissue.mf", "cancer"] if c in df.columns]
    df_final = df.drop(columns=drop_cols)

    df_final.to_csv(out_path, index=False)
    print(f"\n✅ ذخیره شد: {out_path}  (شکل: {df_final.shape[0]} نمونه × {df_final.shape[1]-1} ژن + ۱ ستون کلاس)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("استفاده: python make_leukemia_4c.py golub.csv Leukemia_4c.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
