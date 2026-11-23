"""
تولید سه نمودار SHAP برای بخش جدید "Model Explainability" مقاله — دقیقاً هماهنگ با
همون فیلتری که واقعاً توی پایپ‌لاین استفاده شده (src/shapley_filter.py: RandomForest
با n_estimators=400 و class_weight='balanced'، TreeExplainer).

⚠️ اصلاح نسخه‌ی ۲ (طبق بازخورد استاد): نسخه‌ی قبلی برای نمودار (c) از SHAP
interaction values واقعی (explainer.shap_interaction_values) استفاده می‌کرد. توی
اون نوع ماتریس، قطر اصلی «اثر مستقل» هر ژنه، نه همبستگی خودش با خودش — پس هیچ
دلیلی نداشت حتماً ۱ باشه، که باعث شد نمودار عجیب به‌نظر برسه. الان (مثل خود
chart (1).py که فرستاده بودی) به‌جاش از **ضریب همبستگی پیرسون بین بردارهای SHAP
هر جفت ژن** استفاده می‌شه — قطرش خودکار ۱ می‌شه (همبستگی هر ژن با خودش)، و
مقادیر غیرقطری هم بین -۱ و ۱ معنادار و قابل‌مقایسه‌ان. این کندتر از قبل نیست
(همبستگی محاسبه‌ش آنیه)، برخلاف SHAP interaction که ممکن بود چند دقیقه طول بکشه.

اجرا:
    python generate_shap_explainability.py Lung_Cancer.xlsx

خروجی (توی outputs/):
    shap_fig5_bar_<dataset>.png              — Figure 5 (بار پلات، اهمیت کلی)
    shap_fig6_beeswarm_<dataset>.png          — Figure 6 (جهت و بزرگی اثر)
    shap_fig7_interaction_heatmap_<dataset>.png — Figure 7 (ماتریس همبستگی SHAP)
    shap_top_genes_<dataset>.txt              — لیست اسم ۱۰ ژن برتر (برای نوشتن متن نتایج)
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_and_clean_data

# ===================== تنظیمات (هماهنگ با src/shapley_filter.py) =====================
N_ESTIMATORS = 400          # همون عددی که shapley_filter.py استفاده می‌کنه
TOP_N_FILTER = 60           # همون N فیلتر اصلی مقاله (Table 6)
TOP_N_DISPLAY = 10          # چند تا ژن توی نمودارها نشون داده بشه (خوانا بمونه)
INTERACTION_ON_TOP_N = 60   # از بین همین تعداد، interaction واقعی محاسبه می‌شه؛
                             # اگه کند بود کمش کن (مثلاً 20)
SEED = 42


def main():
    if len(sys.argv) != 2:
        print("استفاده: python generate_shap_explainability.py Lung_Cancer.xlsx")
        sys.exit(1)

    filename = sys.argv[1]
    filepath = f"data/{filename}"
    if not os.path.exists(filepath):
        print(f"❌ فایل پیدا نشد: {filepath}")
        sys.exit(1)

    print(f"📂 در حال بارگذاری {filename} ...")
    X, y, classes = load_and_clean_data(filepath)
    print(f"   شکل داده: {X.shape}, کلاس‌ها: {list(classes)}")

    # ===================== Phase 1 دقیقاً مثل shapley_filter.py =====================
    print(f"🌲 آموزش RandomForest ({N_ESTIMATORS} درخت، class_weight=balanced) روی کل ژن‌ها ...")
    rf_full = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=SEED, class_weight="balanced"
    )
    rf_full.fit(X, y)

    print("🔍 محاسبه‌ی SHAP روی کل ژن‌ها (برای رتبه‌بندی، مثل خود فیلتر) ...")
    explainer_full = shap.TreeExplainer(rf_full)
    shap_values_full = explainer_full.shap_values(X)

    def to_mean_abs(sv):
        if isinstance(sv, list):
            return np.abs(np.array(sv)).mean(axis=(0, 1))
        elif sv.ndim == 3:
            return np.abs(sv).mean(axis=(0, 2))
        return np.abs(sv).mean(axis=0)

    mean_abs_full = to_mean_abs(shap_values_full)
    order_full = np.argsort(-mean_abs_full)
    top_filter_idx = order_full[:min(TOP_N_FILTER, X.shape[1])]
    top_filter_genes = X.columns[top_filter_idx].tolist()

    # ===================== زیرمجموعه‌ی فیلترشده (N=60) — دقیقاً همون چیزی که wrapper می‌بینه =====================
    X_filtered = X[top_filter_genes]

    print(f"🌲 آموزش دوباره روی {len(top_filter_genes)} ژن فیلترشده (برای نمودارهای explainability) ...")
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=SEED, class_weight="balanced"
    )
    rf.fit(X_filtered, y)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_filtered)
    sv_for_rank = to_mean_abs(shap_values)  # برای رتبه‌بندی top-10 نمایشی

    # مقادیر خام SHAP برای beeswarm (نه mean-abs) — کلاس مثبت/آخر رو انتخاب می‌کنیم
    def to_raw_for_class(sv, n_classes):
        if isinstance(sv, list):
            idx = min(1, len(sv) - 1)
            return np.array(sv[idx])
        elif sv.ndim == 3:
            idx = min(1, sv.shape[2] - 1)
            return sv[:, :, idx]
        return sv

    n_classes = len(classes)
    sv_raw = to_raw_for_class(shap_values, n_classes)

    top_display_idx = np.argsort(-sv_for_rank)[:TOP_N_DISPLAY]
    top_display_genes = [top_filter_genes[i] for i in top_display_idx]

    os.makedirs("outputs", exist_ok=True)
    tag = filename.replace(".", "_")

    # ذخیره‌ی اسم ژن‌های برتر برای نوشتن متن نتایج
    with open(f"outputs/shap_top_genes_{tag}.txt", "w", encoding="utf-8") as f:
        f.write("Top genes by mean |SHAP value| (for writing the Results text):\n")
        for rank, gi in enumerate(top_display_idx, 1):
            f.write(f"{rank}. {top_filter_genes[gi]}  (mean|SHAP|={sv_for_rank[gi]:.5f})\n")
    print(f"💾 لیست ژن‌های برتر: outputs/shap_top_genes_{tag}.txt")

    # ===================== Figure 5: Bar plot =====================
    print("🎨 رسم Figure 5 (bar plot) ...")
    plt.figure(figsize=(9, 6))
    order = np.argsort(sv_for_rank[top_display_idx])
    plt.barh(
        [top_display_genes[i] for i in order],
        [sv_for_rank[top_display_idx][i] for i in order],
        color="#2E5B8A",
    )
    plt.xlabel("mean(|SHAP value|)")
    plt.title("(a) SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(f"outputs/shap_fig5_bar_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ===================== Figure 6: Beeswarm (summary plot واقعی از خود shap) =====================
    print("🎨 رسم Figure 6 (beeswarm/summary plot) ...")
    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        sv_raw[:, top_display_idx], X_filtered.iloc[:, top_display_idx],
        feature_names=top_display_genes, show=False,
    )
    plt.title("(b) SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(f"outputs/shap_fig6_beeswarm_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ===================== Figure 7: SHAP value correlation heatmap =====================
    # ⚠️ اصلاح مهم: نسخه‌ی قبلی این اسکریپت از explainer.shap_interaction_values (شاخص
    # تعامل Shapley واقعی) استفاده می‌کرد. توی اون نوع ماتریس، قطر اصلی «اثر مستقل» هر
    # ژنه (نه همبستگی خودش با خودش)، و چون واحدش خودِ SHAP است نه یک عدد نرمال‌شده،
    # هیچ اجباری نداره که ۱ باشه. طبق بازخورد استاد، اینجا به‌جاش از **ضریب همبستگی
    # پیرسون بین بردارهای SHAP هر جفت ژن** استفاده می‌کنیم — دقیقاً کاری که خود
    # chart (1).py هم می‌کرد — که ذاتاً قطرش ۱ می‌شه (همبستگی هر ژن با خودش) و
    # مقادیر غیرقطری هم بین -۱ و ۱ معنادار و قابل‌مقایسه‌ان.
    print("🔬 محاسبه‌ی ضریب همبستگی بین بردارهای SHAP ژن‌های برتر ...")
    sv_top10 = sv_raw[:, top_display_idx]  # (n_samples, 10)
    inter_mat = np.corrcoef(sv_top10.T)    # (10, 10), قطر = ۱ خودکار

    print("🎨 رسم Figure 7 (correlation heatmap) ...")
    plt.figure(figsize=(9, 8))
    sns.heatmap(
        inter_mat, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
        xticklabels=top_display_genes, yticklabels=top_display_genes,
        cbar_kws={"label": "Pearson correlation between SHAP value vectors"},
    )
    plt.title("(c) SHAP Value Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"outputs/shap_fig7_interaction_heatmap_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("\n✅ تمام شد! سه نمودار + لیست ژن‌های برتر توی outputs/ ذخیره شدن.")
    print("   این‌ها رو برام بفرست تا متن بخش‌های ۳.۸ و ۴.۵ رو با اسم ژن‌های واقعی کامل کنم.")


if __name__ == "__main__":
    main()
