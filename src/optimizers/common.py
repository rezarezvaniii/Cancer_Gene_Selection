"""
ابزارهای مشترک بین بهینه‌سازهای HHO و IGWO:
- ساخت طبقه‌بند SVM با تنظیمات مقاله یا انتخاب خودکار کرنل
- کراس‌ولیدیشن تطبیقی (متناسب با کوچیک‌ترین کلاس دیتاست)
- کش فیتنس (برای جلوگیری از محاسبه‌ی دوباره‌ی زیرمجموعه‌های تکراری)
- ارزیابی موازی جمعیت با joblib
"""

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.svm import SVC
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')


def adaptive_cv(y, max_splits=10):
    """تعداد fold ها رو بر اساس کوچیک‌ترین کلاس تنظیم می‌کنه تا خطا نده (مهم برای دیتاست‌های خیلی کوچیک)."""
    _, counts = np.unique(y, return_counts=True)
    n_splits = max(2, min(max_splits, int(counts.min())))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def make_svm(kernel="rbf", C=10.0, gamma="scale", degree=3):
    return SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, random_state=42, tol=1e-4)


def auto_select_kernel(X, y, candidates=None, verbose=True):
    """
    قبل از شروع بهینه‌سازی، روی همون ژن‌های فیلتر شده با kSV، چند کرنل رو
    با یک CV سریع امتحان می‌کنه و بهترین رو برمی‌گردونه.
    این کار جایگزین حدس‌زدن دستی کرنل (poly طبق جدول مقاله) می‌شه و
    باعث می‌شه کد روی هر دیتاستی خودش بهترین تنظیم رو پیدا کنه.
    """
    if candidates is None:
        candidates = [
            {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
            {"kernel": "poly", "C": 10.0, "gamma": 0.001, "degree": 3},
            {"kernel": "linear", "C": 1.0, "gamma": "scale"},
        ]
    cv = adaptive_cv(y, max_splits=5)
    best_score, best_cfg = -np.inf, candidates[0]
    for cfg in candidates:
        clf = make_svm(**cfg)
        try:
            score = cross_val_score(clf, X, y, cv=cv, n_jobs=1).mean()
        except Exception:
            score = -np.inf
        if verbose:
            print(f"   ⚙️  kernel={cfg['kernel']:<7} gamma={cfg.get('gamma')}\tCV acc={score:.4f}")
        if score > best_score:
            best_score, best_cfg = score, cfg
    if verbose:
        print(f"   ✅ Auto-selected kernel: {best_cfg['kernel']} (CV acc={best_score:.4f})")
    return best_cfg


class FitnessCache:
    """کش می‌کنه تا فیتنس یک زیرمجموعه‌ی ژنی که قبلاً محاسبه شده دوباره محاسبه نشه.
    در HHO/IGWO خیلی از شاهین‌ها/گرگ‌ها بعد از باینری‌سازی به یک زیرمجموعه‌ی یکسان می‌رسند."""

    def __init__(self, X, y, svm_cfg, penalty_coef, n_features_total):
        self.X = X
        self.y = y
        self.svm_cfg = svm_cfg
        self.penalty_coef = penalty_coef
        self.n_features_total = n_features_total
        self.cache = {}
        self.cv = adaptive_cv(y)

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