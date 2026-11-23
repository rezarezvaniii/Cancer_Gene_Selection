"""
ابزارهای مشترک بین بهینه‌سازهای HHO و IGWO:
- ساخت طبقه‌بند SVM با تنظیمات مقاله یا انتخاب خودکار کرنل
- کراس‌ولیدیشن تطبیقی (متناسب با کوچیک‌ترین کلاس دیتاست)
- کش فیتنس (برای جلوگیری از محاسبه‌ی دوباره‌ی زیرمجموعه‌های تکراری)
- ارزیابی موازی جمعیت با joblib
"""

import math
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.svm import SVC
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')


def levy_flight(shape, rng, beta=1.5):
    """پرواز لِوی (الگوریتم Mantegna) — برای SBOA (مرحله‌ی سوم اکتشاف، Eq. 5 مقاله).
    beta=1.5: مقدار استاندارد استفاده‌شده در اکثر مقالات متاهیوریستیک (از جمله SBOA اصلی)."""
    num = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma_u = (num / den) ** (1 / beta)
    u = rng.normal(0, sigma_u, shape)
    v = rng.normal(0, 1, shape)
    return u / (np.abs(v) ** (1 / beta))


def adaptive_n_splits(y, max_splits=10):
    """تعداد fold ها رو بر اساس کوچیک‌ترین کلاس تنظیم می‌کنه تا خطا نده (مهم برای دیتاست‌های خیلی کوچیک)."""
    _, counts = np.unique(y, return_counts=True)
    return max(2, min(max_splits, int(counts.min())))


def adaptive_cv(y, max_splits=10):
    return StratifiedKFold(n_splits=adaptive_n_splits(y, max_splits), shuffle=True, random_state=42)


def make_svm(kernel="rbf", C=10.0, gamma="scale", degree=3, class_weight=None):
    return SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, class_weight=class_weight,
               random_state=42, tol=1e-4)


def auto_select_kernel(X, y, candidates=None, verbose=True, n_repeats=3):
    """
    قبل از شروع بهینه‌سازی، روی همون ژن‌های فیلتر شده با kSV، چند ترکیب
    kernel/C/class_weight رو با یک CV تکرارشونده امتحان می‌کنه و بهترین رو برمی‌گردونه.

    نسبت به نسخه‌ی قبلی دو تغییر مهم:
      1) هم با class_weight=None و هم class_weight='balanced' امتحان می‌شه.
         دیتاست‌هایی مثل Colon (40 تومور / 22 نرمال) نامتوازن هستن و SVM بدون
         class_weight گرایش به کلاس اکثریت پیدا می‌کنه که دقت رو پایین میاره.
      2) چند مقدار C هم امتحان می‌شه (نه فقط یک عدد ثابت حدسی)، چون بهترین C
         بین دیتاست‌ها فرق می‌کنه.
      3) به‌جای یک StratifiedKFold ثابت، میانگین چند تکرار گرفته می‌شه تا
         انتخاب کرنل به یک split خاص و شانسی وابسته نباشه (مهم برای دیتاست‌های
         خیلی کوچیک مثل CNS با ۶۰ نمونه).
    """
    if candidates is None:
        candidates = []
        for cw in (None, "balanced"):
            for C in (1.0, 10.0, 100.0):
                candidates.append({"kernel": "rbf", "C": C, "gamma": "scale", "class_weight": cw})
                candidates.append({"kernel": "linear", "C": C, "gamma": "scale", "class_weight": cw})
            candidates.append({"kernel": "poly", "C": 10.0, "gamma": 0.001, "degree": 3, "class_weight": cw})

    best_score, best_cfg = -np.inf, candidates[0]
    for cfg in candidates:
        scores = []
        for r in range(n_repeats):
            cv = StratifiedKFold(n_splits=adaptive_cv(y, max_splits=5).n_splits,
                                  shuffle=True, random_state=100 + r)
            clf = make_svm(**cfg)
            try:
                scores.append(cross_val_score(clf, X, y, cv=cv, n_jobs=1).mean())
            except Exception:
                scores.append(-np.inf)
        score = float(np.mean(scores))
        if verbose:
            cw_tag = cfg.get("class_weight") or "none"
            print(f"   ⚙️  kernel={cfg['kernel']:<7} C={cfg.get('C'):<6} gamma={cfg.get('gamma')} "
                  f"class_weight={cw_tag:<8}\tCV acc={score:.4f}")
        if score > best_score:
            best_score, best_cfg = score, cfg
    if verbose:
        print(f"   ✅ Auto-selected config: {best_cfg} (CV acc={best_score:.4f})")
    return best_cfg


class FitnessCache:
    """کش می‌کنه تا فیتنس یک زیرمجموعه‌ی ژنی که قبلاً محاسبه شده دوباره محاسبه نشه.
    در IGWO خیلی از گرگ‌ها بعد از باینری‌سازی به یک زیرمجموعه‌ی یکسان می‌رسند.

    ⚠️ BUGFIX مهم (overfitting به CV): نسخه‌ی قبلی از یک StratifiedKFold *ثابت* (همون
    یک تقسیم‌بندی، با random_state=42) برای ارزیابی فیتنس همه‌ی کاندیدها طی کل جستجو
    استفاده می‌کرد. چون طی جستجو ده‌ها هزار زیرمجموعه‌ی یکتا امتحان می‌شه (روی Colon مثلاً
    ۳۸۷۳۸ تا)، این یعنی الگوریتم عملاً داره دنبال زیرمجموعه‌ای می‌گرده که *دقیقاً روی همون
    یک تقسیم‌بندی* بهترین امتیاز رو بگیره — که با دیتاست‌های کوچیک (Colon=62، CNS=60 نمونه)
    یه سوگیری آماری کلاسیکه (به آزمایش نشون داده شد: فیتنس جستجو ۰.۹۱۷۹ می‌شد ولی ارزیابی
    نهایی دقیق فقط ۸۵.۸۷٪ در می‌اومد، یعنی زیرمجموعه‌ی «برنده» فقط شانسی رو همون split
    خوب بوده، نه واقعاً تعمیم‌پذیر). راه‌حل: به‌جای یک fold ثابت، از RepeatedStratifiedKFold
    (چند تکرار تقسیم‌بندی متفاوت) میانگین گرفته می‌شه تا فیتنس جستجو به تعمیم‌پذیری واقعی
    نزدیک‌تر باشه، نه نویز یک split خاص.

    cv_splits/cv_repeats: تعداد fold و تکرار طی *جستجو* (نه ارزیابی نهایی). عدد نهایی
    گزارش‌شده (Highest/Average/Lowest) هنوز جدا با evaluate_final_subset و
    RepeatedStratifiedKFold کامل (تا ۱۰ fold، ۳۰ تکرار، دقیقاً مثل مقاله) محاسبه می‌شه."""

    def __init__(self, X, y, svm_cfg, penalty_coef, n_features_total, cv_splits=5, cv_repeats=2):
        self.X = X
        self.y = y
        self.svm_cfg = svm_cfg
        self.penalty_coef = penalty_coef
        self.n_features_total = n_features_total
        self.cache = {}
        n_splits = adaptive_n_splits(y, max_splits=cv_splits)
        self.cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=cv_repeats, random_state=42)

    def _score_one(self, key):
        if key in self.cache:
            return self.cache[key]
        indices = list(key)
        if len(indices) < 2:
            self.cache[key] = 0.0
            return 0.0
        X_subset = self.X.iloc[:, indices]
        clf = make_svm(**self.svm_cfg)
        try:
            scores = cross_val_score(clf, X_subset, self.y, cv=self.cv, n_jobs=1)
            acc = scores.mean()
            penalty = self.penalty_coef * (len(indices) / self.n_features_total)
            fitness = acc - penalty
        except Exception:
            fitness = 0.0
        self.cache[key] = fitness
        return fitness

    def batch_score(self, binary_positions, n_jobs=-1):
        """binary_positions: آرایه (N, n_features) صفر و یک. خروجی: آرایه فیتنس‌ها."""
        keys = [tuple(sorted(np.where(row == 1)[0].tolist())) for row in binary_positions]
        uncached = [k for k in set(keys) if k not in self.cache]
        if uncached:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(self._score_one)(k) for k in uncached
            )
            for k, r in zip(uncached, results):
                self.cache[k] = r
        return np.array([self.cache[k] for k in keys])

    def raw_accuracy(self, indices):
        """دقت خام (بدون جریمه) برای گزارش نهایی."""
        if len(indices) < 2:
            return 0.0
        X_subset = self.X.iloc[:, indices]
        clf = make_svm(**self.svm_cfg)
        try:
            return cross_val_score(clf, X_subset, self.y, cv=self.cv, n_jobs=1).mean()
        except Exception:
            return 0.0


def evaluate_topk_subset(X, y, ranked_genes, k, svm_cfg, n_repeats=10, n_splits=None):
    """
    برای درخواست استاد: «همون تعداد ژنی که مقاله گزارش کرده رو بهش بده، ببین با همون
    تعداد می‌تونه نتیجه‌ی خوب پیدا کنه یا نه». این تابع دقیقاً K تا از بالای لیست
    ranked_genes (خروجی get_kSV_features، مرتب بر اساس اهمیت SHAP) رو برمی‌داره و
    مثل evaluate_final_subset ارزیابی می‌کنه — بدون اینکه IGWO مجبور باشه خودش به K
    ژن برسه. اگه نتیجه هنوز ضعیف بود، یعنی مشکل از «تعداد» ژن نیست، از «کدوم ژن‌ها»ست.
    """
    k = max(2, min(k, len(ranked_genes)))
    top_k_genes = ranked_genes[:k]
    result = evaluate_final_subset(X, y, top_k_genes, svm_cfg, n_repeats=n_repeats, n_splits=n_splits)
    result["genes"] = top_k_genes
    result["k"] = k
    return result


def evaluate_final_subset(X, y, genes, svm_cfg, n_repeats=5, n_splits=None):
    """
    ارزیابی نهایی زیرمجموعه‌ی انتخاب‌شده به سبک جداول مقاله:
    Highest / Average / Lowest روی چند تکرار مستقل از کراس‌ولیدیشن.
    """
    if len(genes) < 2:
        return {"highest": 0.0, "average": 0.0, "lowest": 0.0}

    X_subset = X[genes]
    if n_splits is None:
        _, counts = np.unique(y, return_counts=True)
        n_splits = max(2, min(10, int(counts.min())))

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)
    clf = make_svm(**svm_cfg)
    fold_scores = cross_val_score(clf, X_subset, y, cv=rskf, n_jobs=-1)

    # میانگین هر "تکرار" (repeat) رو جدا حساب می‌کنیم تا شبیه ۳۰-run مقاله بشه
    per_repeat = fold_scores.reshape(n_repeats, n_splits).mean(axis=1)
    return {
        "highest": float(per_repeat.max()),
        "average": float(per_repeat.mean()),
        "lowest": float(per_repeat.min()),
    }