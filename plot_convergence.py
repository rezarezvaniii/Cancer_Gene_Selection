"""
رسم نمودار «Accuracy بر حسب Iteration» و «Loss بر حسب Iteration» حین آموزش
(یعنی حین جستجوی بهینه‌سازی، نه بعدش) روی Leukemia-3c یا Leukemia-4c، برای
چند الگوریتم کنار هم — دقیقاً همون خواسته‌ی استاد.

⚠️ نکته‌ی مهم درباره‌ی این‌که «loss» و «accuracy» اینجا چی هستن:
این یه الگوریتم متاهیوریستیک (RIME-LF/IGWO/...) است، نه شبکه‌ی عصبی — پس «loss»
به‌معنای معمول (خروجی مستقیم یک تابع loss قابل‌مشتق‌گیری) وجود نداره. طبق قرارداد
رایج در همین حوزه (مقاله‌های IGWO/RIME/HDRIME و مشابه)، «accuracy» = بهترین فیتنس
(alpha_score) تا اون ایتریشن (که چون penalty_coef خیلی کوچیکه، تقریباً همون دقت
CV خامه)، و «loss» = 1 - accuracy (نرخ خطا). این دقیقاً همون چیزیه که در
convergence curve این‌جور مقاله‌ها رسم می‌شه.

⚠️ برای این‌که منحنی‌ها واقعاً قابل‌مقایسه باشن (طول یکسان، بدون قطع زودهنگام)،
patience رو خیلی بزرگ گذاشتم (early stopping خاموش) — وگرنه اگه یه الگوریتم
زودتر از بقیه early-stop کنه، نمودارش کوتاه‌تر می‌شه و مقایسه بصری گمراه‌کننده
می‌شه.

اجرا:
    python plot_convergence.py Leukemia_3c.csv
    یا
    python plot_convergence.py Leukemia_4c.csv
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import load_and_clean_data
from src.shapley_filter import get_kSV_features
from src.optimizers.common import auto_select_kernel
from src.optimizers.rime_lf_optimizer import RIMELF_Optimizer
from src.optimizers.igwo_optimizer import IGWO_Optimizer
from src.optimizers.igo_optimizer import IGO_Optimizer
from src.optimizers.isboa_optimizer import ISBOA_Optimizer

# ===================== تنظیمات =====================
TOP_N_SHAPLEY = 60
N_AGENTS = 60          # کمتر از اجرای نهایی گذاشتم تا این نمودار در زمان معقولی دربیاد؛
                        # اگه می‌خوای دقیقاً با تنظیمات اصلی main.py یکی باشه، برابر همونی
                        # که آنجا گذاشتی کن (N_AGENTS اونجا رو چک کن)
MAX_ITER = 100
PENALTY_COEF = 0.001
SEARCH_CV_SPLITS = 5
SEARCH_CV_REPEATS = 2
SEED = 42

# الگوریتم‌هایی که می‌خوایم روی هم نمودار بدیم — کلید = اسمی که توی legend میاد
ALGORITHMS_TO_RUN = {
    "RIME-LF (proposed)": RIMELF_Optimizer,
    "IGWO (baseline)": IGWO_Optimizer,
    "IGO": IGO_Optimizer,
    "ISBOA": ISBOA_Optimizer,
}


def main():
    if len(sys.argv) != 2:
        print("استفاده: python plot_convergence.py Leukemia_3c.csv   (یا Leukemia_4c.csv)")
        sys.exit(1)

    filename = sys.argv[1]
    filepath = f"data/{filename}"
    if not os.path.exists(filepath):
        print(f"❌ فایل پیدا نشد: {filepath}")
        sys.exit(1)

    print(f"📂 در حال بارگذاری {filename} ...")
    X, y, classes = load_and_clean_data(filepath)

    print("🔍 فیلتر SHAP...")
    top_genes, _ = get_kSV_features(X, y, top_n=TOP_N_SHAPLEY)
    X_reduced = X[top_genes]

    print("⚙️  انتخاب خودکار کرنل SVM (مشترک بین همه‌ی الگوریتم‌ها، برای مقایسه‌ی منصفانه)...")
    svm_cfg = auto_select_kernel(X_reduced, y, verbose=False)
    print(f"   کرنل انتخاب‌شده: {svm_cfg}")

    histories = {}
    for name, OptClass in ALGORITHMS_TO_RUN.items():
        print(f"\n▶️  در حال اجرای {name} ...")
        optimizer = OptClass(
            X_reduced, y,
            svm_cfg=svm_cfg,
            n_wolves=N_AGENTS,
            max_iter=MAX_ITER,
            penalty_coef=PENALTY_COEF,
            n_jobs=-1,
            verbose=True,
            seed=SEED,
            patience=MAX_ITER + 1,   # ⚠️ early stopping خاموش — برای این‌که منحنی‌ها هم‌طول بمونن
            search_cv_splits=SEARCH_CV_SPLITS,
            search_cv_repeats=SEARCH_CV_REPEATS,
        )
        _, _, history, _ = optimizer.optimize()
        histories[name] = np.array(history)
        print(f"   ✔️  {name}: {len(history)} ایتریشن، فیتنس نهایی={history[-1]:.4f}")

    os.makedirs("outputs", exist_ok=True)

    # ===================== نمودار ۱: Accuracy بر حسب Iteration =====================
    plt.figure(figsize=(9, 6))
    for name, hist in histories.items():
        plt.plot(range(1, len(hist) + 1), hist, label=name, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness (\u2248 Accuracy)")
    plt.title(f"Accuracy vs. Iteration \u2014 {filename}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    acc_path = f"outputs/convergence_accuracy_{filename.replace('.', '_')}.png"
    plt.savefig(acc_path, dpi=150)
    print(f"\n💾 ذخیره شد: {acc_path}")

    # ===================== نمودار ۲: Loss (=1-Accuracy) بر حسب Iteration =====================
    plt.figure(figsize=(9, 6))
    for name, hist in histories.items():
        loss = 1.0 - hist
        plt.plot(range(1, len(loss) + 1), loss, label=name, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (1 \u2212 Best Fitness)")
    plt.title(f"Loss vs. Iteration \u2014 {filename}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    loss_path = f"outputs/convergence_loss_{filename.replace('.', '_')}.png"
    plt.savefig(loss_path, dpi=150)
    print(f"💾 ذخیره شد: {loss_path}")


if __name__ == "__main__":
    main()
