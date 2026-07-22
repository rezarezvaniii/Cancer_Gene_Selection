"""
لایه‌ی هماهنگ‌کننده برای هر دیتاست:
  1) بهترین کرنل SVM رو پیدا می‌کنه (شروع از poly طبق مقاله، ولی rbf/linear هم امتحان می‌شن
     تا اگه چیزی بهتر از مقاله پیدا شد از دستمون در نره)
  2) IGWO رو n_runs بار با seed های متفاوت اجرا می‌کنه (فقط IGWO — طبق مقاله، نه HHO)
  3) بهترین run رو با ارزیابی نهایی (Highest/Average/Lowest, سبک مقاله) انتخاب می‌کنه
"""

from .optimizers.common import auto_select_kernel, evaluate_final_subset
from .optimizers.igwo_optimizer import IGWO_Optimizer


def run_pipeline(X_reduced, y, dataset_name,
                  n_runs=3,
                  n_agents=150,
                  max_iter=200,
                  penalty_coef=0.001,
                  final_eval_repeats=10,
                  n_jobs=-1,
                  verbose=True,
                  patience=30):

    print(f"\n🔬 Selecting SVM kernel for {dataset_name}...")
    svm_cfg = auto_select_kernel(X_reduced, y, verbose=verbose)

    all_results = []

    for run_idx in range(n_runs):
        seed = 1000 * (run_idx + 1)
        print(f"\n▶️  [{dataset_name}] IGWO — run {run_idx + 1}/{n_runs}")

        optimizer = IGWO_Optimizer(
            X_reduced, y,
            svm_cfg=svm_cfg,
            n_wolves=n_agents,
            max_iter=max_iter,
            penalty_coef=penalty_coef,
            n_jobs=n_jobs,
            verbose=verbose,
            seed=seed,
            patience=patience,
        )
        genes, fitness, history = optimizer.optimize()

        final_metrics = evaluate_final_subset(
            X_reduced, y, genes, svm_cfg, n_repeats=final_eval_repeats
        )

        all_results.append({
            "algorithm": "IGWO",
            "run": run_idx + 1,
            "genes": genes,
            "n_genes": len(genes),
            "fitness_during_search": fitness,
            "history": history,
            "highest": final_metrics["highest"],
            "average": final_metrics["average"],
            "lowest": final_metrics["lowest"],
            "svm_cfg": svm_cfg,
        })

        print(f"   ✔️  run {run_idx + 1}: "
              f"avg_acc={final_metrics['average']*100:.2f}% "
              f"(high={final_metrics['highest']*100:.2f}%, low={final_metrics['lowest']*100:.2f}%) "
              f"| genes={len(genes)}")

    # بهترین run: اول بر اساس average accuracy، در تساوی بر اساس تعداد ژن کمتر
    best = max(all_results, key=lambda r: (round(r["average"], 6), -r["n_genes"]))

    return {
        "dataset": dataset_name,
        "svm_cfg": svm_cfg,
        "best": best,
        "all_results": all_results,
    }