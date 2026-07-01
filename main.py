from src.data_loader import load_and_clean_data
from src.shapley_filter import get_kSV_features
from src.igwo_optimizer import IGWO_Optimizer
from src.visualizer import plot_results # اضافه شد

def main():
    # 1. لود داده
    X, y, classes = load_and_clean_data('data/golub.csv')
    
    # 2. فاز شاپلی
    top_n_genes = 50
    top_genes, importance_df = get_kSV_features(X, y, top_n=top_n_genes)
    X_reduced = X[top_genes]
    
    # 3. فاز IGWO
    optimizer = IGWO_Optimizer(X_reduced, y, n_wolves=20, max_iter=30)
    best_genes, best_acc, history = optimizer.optimize() # خروجی history اضافه شد
    
    # 4. نتایج نهایی
    print("\n" + "="*30)
    print("FINAL RESULTS")
    print(f"Best Accuracy: {best_acc*100:.2f}%")
    print(f"Number of Selected Genes: {len(best_genes)}")
    print("="*30)
    
    # 5. رسم نمودار
    plot_results(history, len(best_genes), best_acc)

if __name__ == "__main__":
    main()