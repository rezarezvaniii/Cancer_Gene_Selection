import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(history, selected_genes_count, best_acc):
    # تنظیم استایل نمودار
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    # 1. نمودار همگرایی (Convergence Curve)
    ax.plot(range(1, len(history) + 1), history, marker='o', color='b', linestyle='-', linewidth=2)
    ax.set_title(f'IGWO Convergence Curve\n(Final Accuracy: {best_acc*100:.2f}% | Genes: {selected_genes_count})')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness Score (Accuracy - Penalty)')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('results_plot.png') # ذخیره نمودار در یک فایل
    print("\nGraph saved as 'results_plot.png'")
    plt.show()