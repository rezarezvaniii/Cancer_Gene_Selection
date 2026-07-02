from src.data_loader import load_and_clean_data
from src.shapley_filter import get_kSV_features
from src.igwo_optimizer import IGWO_Optimizer
from src.visualizer import plot_results
import pandas as pd

def main():
    # 1. لیست تمام دیتاست‌هایی که دانلود کردی را اینجا بنویس
    dataset_files = ["Colon.arff"] # مثلا اگر بعدی را دانلود کردی اضافه کن: ["Colon.arff", "SRBCT.arff"]
    
    final_summary = []

    for filename in dataset_files:
        print(f"\n" + ">>>" * 10)
        print(f"PROCESSING DATASET: {filename}")
        
        try:
            # بارگذاری داده
            X, y, classes = load_and_clean_data(f'data/{filename}')
            
            # فاز 1: فیلتر شاپلی
            top_genes, _ = get_kSV_features(X, y, top_n=100)
            X_reduced = X[top_genes]
            
            # فاز 2: بهینه‌سازی IGWO
            # تعداد گرگ و تکرار را برای دقت بیشتر بالا بردیم
            optimizer = IGWO_Optimizer(X_reduced, y, n_wolves=20, max_iter=80)
            best_genes, best_acc, history = optimizer.optimize()
            
            # ذخیره نتیجه برای گزارش نهایی
            final_summary.append({
                "Dataset": filename,
                "Accuracy": f"{best_acc*100:.2f}%",
                "Genes Count": len(best_genes)
            })

            # رسم نمودار برای هر دیتاست
            plot_results(history, len(best_genes), best_acc)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # نمایش جدول نهایی در کنسول
    print("\n" + "="*40)
    print("FINAL BENCHMARK SUMMARY")
    print(pd.DataFrame(final_summary))
    print("="*40)

if __name__ == "__main__":
    main()