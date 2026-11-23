"""
لایه‌ی هماهنگ‌کننده برای هر دیتاست:
  1) بهترین کرنل SVM رو پیدا می‌کنه (شروع از poly طبق مقاله، ولی rbf/linear هم امتحان می‌شن
     تا اگه چیزی بهتر از مقاله پیدا شد از دستمون در نره)
  2) RIME-LF (یا هر الگوریتم دیگه‌ای که انتخاب شده) رو n_runs بار با seed های متفاوت اجرا می‌کنه
  3) هر run دقیقاً یک عدد نهایی می‌ده (بدون double-dipping — پایین توضیح داده شده)، و عدد
     سرتیتری که گزارش می‌شه Highest/Average/Lowest بین این n_runs اجراست — دقیقاً سبک
     مقاله‌ی پایه («هر روش N بار اجرا شد و میانگین گزارش شد»)

⚠️ اصلاح مهم (اوت ۲۰۲۶، بعد از بازبینی روش‌شناسی): نسخه‌ی قبلی این فایل توی هر run،
از بین چند کاندید برتر (n_top_candidates)، هرکدوم رو که در ارزیابی نهایی (evaluate_final_subset)
بیشترین "average" رو می‌گرفت انتخاب می‌کرد — یعنی همون معیاری که قراره گزارش بشه، برای
انتخاب هم استفاده می‌شد (double-dipping / انتخاب بهترین از بین چند گزینه با همون معیار
گزارش). این عدد نهایی رو خوش‌بینانه‌تر از واقعیت نشون می‌ده. راه‌حل: کاندید داخل هر run
با معیار مستقل (فیتنس جستجو، که از یک نمونه‌گیری CV متفاوت میاد) انتخاب می‌شه، نه با
همون معیار ارزیابی نهایی. سپس، به‌جای انتخاب "بهترین run" از بین n_runs (که این خودش
هم یه‌جور همون مشکل رو در سطح run تکرار می‌کرد)، میانگین/بیشینه/کمینه بین n_runs مستقل
گزارش می‌شه — این دقیقاً همون چیزیه که پروتکل مقاله‌ی پایه می‌گه.
"""

import os
import json
import numpy as np

from .optimizers.common import auto_select_kernel, evaluate_final_subset
from .optimizers.igwo_optimizer import IGWO_Optimizer
from .optimizers.rime_lf_optimizer import RIMELF_Optimizer
from .optimizers.igo_optimizer import IGO_Optimizer
from .optimizers.isboa_optimizer import ISBOA_Optimizer

ALGORITHMS = {
    "IGWO": IGWO_Optimizer,
    "RIME-LF": RIMELF_Optimizer,
    "IGO": IGO_Optimizer,
    "ISBOA": ISBOA_Optimizer,
}


def run_pipeline(X_reduced, y, dataset_name,
                  n_runs=3,
                  n_agents=150,
                  max_iter=200,
                  penalty_coef=0.001,
                  final_eval_repeats=10,
                  n_jobs=-1,
                  verbose=True,
                  patience=30,
                  search_cv_splits=5,
                  search_cv_repeats=2,
                  n_top_candidates=3,
                  algorithm="IGWO",
                  checkpoint_dir=None):

    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm نامعتبر: {algorithm!r} — یکی از {list(ALGORITHMS)} باشه")
    Optimizer = ALGORITHMS[algorithm]

    print(f"\n🔬 Selecting SVM kernel for {dataset_name}...")
    svm_cfg = auto_select_kernel(X_reduced, y, verbose=verbose)

    # ===== چک‌پوینت سطح run: با ۳۰ اجرا، اگه وسط کار قطع بشه نباید همه‌ی run های
    # تموم‌شده‌ی این دیتاست از دست برن. هر run بعد از تموم شدن فوراً ذخیره می‌شه. =====
    all_results = []
    ckpt_path = None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        safe = str(dataset_name).replace(" ", "_").replace("/", "_")
        ckpt_path = os.path.join(checkpoint_dir, f"runs_{safe}_{algorithm}.json")
        if os.path.exists(ckpt_path):
            try:
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                if saved.get("svm_cfg") == svm_cfg:
                    all_results = saved["runs"][:n_runs]
                    print(f"♻️  {len(all_results)} run از قبل برای {dataset_name} ذخیره شده — ادامه از run {len(all_results) + 1}")
                else:
                    print("⚠️  کرنل SVM با چک‌پوینت قبلی فرق داره — run ها از اول اجرا می‌شن.")
            except Exception as e:
                print(f"⚠️  خوندن چک‌پوینت run ها شکست خورد ({e}) — از اول.")
                all_results = []
    done_runs = {r["run"] for r in all_results}

    for run_idx in range(n_runs):
        if (run_idx + 1) in done_runs:
            continue
        seed = 1000 * (run_idx + 1)
        print(f"\n▶️  [{dataset_name}] {algorithm} — run {run_idx + 1}/{n_runs}")

        optimizer = Optimizer(
            X_reduced, y,
            svm_cfg=svm_cfg,
            n_wolves=n_agents,
            max_iter=max_iter,
            penalty_coef=penalty_coef,
            n_jobs=n_jobs,
            verbose=verbose,
            seed=seed,
            patience=patience,
            search_cv_splits=search_cv_splits,
            search_cv_repeats=search_cv_repeats,
            n_top_candidates=n_top_candidates,
        )
        genes, fitness, history, top_candidates = optimizer.optimize()

        # ✅ انتخاب کاندید فقط با فیتنس جستجو (top_candidates از قبل بر همین اساس مرتبه —
        # رجوع کن به rime_lf_optimizer.py) — نه با ارزیابی نهایی. یعنی معیار انتخاب و معیار
        # گزارش کاملاً مستقل و جدا از هم‌اند، پس دیگه انتخاب "بهترین از بین چندتا با همون
        # معیار گزارش" اتفاق نمی‌افته.
        best_genes, _best_search_fitness = top_candidates[0]
        final_metrics = evaluate_final_subset(
            X_reduced, y, best_genes, svm_cfg, n_repeats=final_eval_repeats
        )
        genes = best_genes

        all_results.append({
            "algorithm": algorithm,
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

        if ckpt_path:
            all_results.sort(key=lambda r: r["run"])
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({"svm_cfg": svm_cfg, "runs": all_results}, f,
                          ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))

        print(f"   ✔️  run {run_idx + 1}: "
              f"avg_acc={final_metrics['average']*100:.2f}% "
              f"(high={final_metrics['highest']*100:.2f}%, low={final_metrics['lowest']*100:.2f}%) "
              f"| genes={len(genes)} (کاندید انتخاب‌شده با بالاترین فیتنس جستجو)")

    # ✅ عدد سرتیتر گزارش = میانگین "average" بین n_runs اجرای مستقل (نه بهترین run) —
    # این دقیقاً همون چیزیه که مقاله‌ی پایه می‌گه: «هر روش N بار اجرا شد، میانگین گزارش شد».
    all_results.sort(key=lambda r: r["run"])
    run_averages = [r["average"] for r in all_results]
    best_run = max(all_results, key=lambda r: r["average"])   # فقط برای گزارش شفافیت، نه انتخاب نهایی
    worst_run = min(all_results, key=lambda r: r["average"])  # همچنین

    # برای تعداد ژن گزارش‌شده، اجرایی که "average"ش نزدیک‌ترین به میانگین کل runs است رو
    # به‌عنوان نماینده انتخاب می‌کنیم (نه بهترین) — نماینده‌ی معمول‌ترین رفتار الگوریتم
    mean_avg = float(np.mean(run_averages))
    representative_run = min(all_results, key=lambda r: abs(r["average"] - mean_avg))

    aggregate = {
        "algorithm": algorithm,
        "n_runs": n_runs,
        "highest": max(run_averages),           # بهترین run (بین n_runs مستقل)
        "average": sum(run_averages) / n_runs,  # ✅ عدد اصلی گزارش — میانگین بین n_runs
        "lowest": min(run_averages),            # بدترین run
        "std": float(np.std(run_averages, ddof=1)) if n_runs > 1 else 0.0,  # انحراف معیار بین run ها
        "run_averages": run_averages,           # همه‌ی n_runs عدد — برای آزمون Wilcoxon
        "mean_n_genes": float(np.mean([r["n_genes"] for r in all_results])),
        "n_genes": representative_run["n_genes"],
        "genes": representative_run["genes"],
        "svm_cfg": svm_cfg,
        # این دو تا فقط برای بررسی شفافیت/دیباگ نگه داشته شدن، نه برای گزارش رسمی مقاله:
        "_best_run_details": best_run,
        "_worst_run_details": worst_run,
    }

    return {
        "dataset": dataset_name,
        "svm_cfg": svm_cfg,
        "best": aggregate,   # ⚠️ اسم کلید "best" برای سازگاری با main.py نگه داشته شده،
                              # ولی محتواش دیگه "بهترین run" نیست — میانگین n_runs مستقله
        "all_results": all_results,
    }