from src.data_loader import load_and_clean_data
from src.shapley_filter import get_kSV_features
# from src.igwo_optimizer import IGWO_Optimizer
from src.hho_optimizer import HHO_Optimizer
from src.visualizer import plot_results
import pandas as pd
import os

def main():
    # 1. لیست تمام دیتاست‌هایی که دانلود کردی (دقیقاً مطابق نام فایل‌هایت)
    dataset_files = [
     "SRBCT.arff",
     "Leukemia.arff",
    ]
    #  "Colon.arff",
    #  "MLL.arff",
    
    final_summary = []

    for filename in dataset_files:
        print(f"\n" + "=" * 50)
        print(f"🚀 PROCESSING DATASET: {filename}")
        print("=" * 50)
        
        # بررسی وجود فایل برای جلوگیری از خطا
        if not os.path.exists(f'data/{filename}'):
            print(f"❌ Error: File 'data/{filename}' not found! Skipping...")
            continue

        try:
            # بارگذاری داده
            X, y, classes = load_and_clean_data(f'data/{filename}')
            
            # فاز 1: فیلتر شاپلی (انتخاب 60 ژن برتر طبق جدول 3 مقاله)
            # این مرحله برای Leukemia و MLL ممکنه چند دقیقه طول بکشه
            top_genes, _ = get_kSV_features(X, y, top_n=60)
            X_reduced = X[top_genes]
            
            # فاز 2: بهینه‌سازی IGWO (تنظیمات طلایی: 100 گرگ و 100 تکرار)
            optimizer = HHO_Optimizer(X_reduced, y, n_hawks=50, max_iter=100)
            best_genes, best_acc, history = optimizer.optimize()
            
            # ذخیره نتیجه برای جدول نهایی
            final_summary.append({
                "Dataset": filename,
                "Final Accuracy": f"{best_acc*100:.2f}%",
                "Selected Genes": len(best_genes)
            })

            # رسم نمودار همگرایی برای این دیتاست
            # نکته: نمودار هر دیتاست جداگانه نمایش داده می‌شود
            plot_results(history, len(best_genes), best_acc)
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    # نمایش جدول نهایی مقایسه‌ای در کنسول (خروجی اصلی پروژه شما)
    print("\n" + "📊" * 15)
    print("   FINAL MULTI-DATASET COMPARISON")
    print("📊" * 15)
    if final_summary:
        summary_df = pd.DataFrame(final_summary)
        print(summary_df.to_string(index=False))
    else:
        print("No results to display.")
    print("=" * 40)

if __name__ == "__main__":
    main()