"""
IGWO (Improved Grey Wolf Optimizer) — پیاده‌سازی وفادار به مقاله
"Feature selection using game Shapley improved grey wolf optimizer" (Afreen et al., 2025)

استراتژی DLH (Dimension Learning-based Hunting) پیاده شده (معادلات ۸ تا ۱۱ مقاله). هر گرگ
دو کاندید موقعیت می‌گیره:
  1) X_i-GWO : آپدیت کلاسیک بر اساس alpha/beta/delta
  2) X_i-DLH : یادگیری از همسایه‌های محلی + یک گرگ تصادفی از کل جمعیت
و هرکدوم فیتنس بهتری داشت انتخاب می‌شه.

⚠️ دو اصلاح مهم نسبت به نسخه‌ی قبلی (هر دو باگ واقعی بودن، نه صرفاً بهینه‌سازی):

1) BUGFIX دقت: نسخه‌ی قبلی در انتهای optimize()، ژن‌های نهایی رو با یک بار *دوباره*
   و *تصادفی* باینری‌کردن alpha_pos می‌ساخت (self._binarize(alpha_pos))، نه از همون
   بردار باینری‌ای که واقعاً به alpha_score رسیده بود. چون _binarize یک تابع تصادفیه
   (rand < sigmoid(x))، این یعنی ژن‌های نهایی هیچ ربطی به فیتنسی که در طول جستجو
   گزارش می‌شد نداشتن! (مثلاً "بهترین فیتنس ۰.۹۱۶۶" لاگ می‌شد ولی ژن‌های برگشتی
   زیرمجموعه‌ای کاملاً متفاوت و کم‌دقت‌تر بودن). این دقیقاً چرایی عقب‌موندن
   Colon/CNS از مقاله در اجرای قبلی بود. حالا alpha_bin (بردار باینری واقعی) در
   کنار alpha_score/alpha_pos ردیابی می‌شه و مستقیماً برای خروجی نهایی استفاده می‌شه.

2) BUGFIX کارایی: نسخه‌ی قبلی برای هر گرگ به‌صورت جدا و سریال (n_jobs=1) کاندیدهاش
   رو ارزیابی می‌کرد (۱۵۰ فراخوانی سریال در هر ایتریشن) که باعث می‌شد از هسته‌های CPU
   موجود عملاً استفاده نشه و اجرا خیلی کند بشه. حالا کل به‌روزرسانی جمعیت (GWO + DLH)
   با numpy وکتورایز شده و فیتنس همه‌ی ۲n کاندید (GWO و DLH برای همه‌ی گرگ‌ها) در
   یک فراخوانی batch موازی (n_jobs=-1) محاسبه می‌شه.
"""

import numpy as np
from .common import FitnessCache


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-10 * (x - 0.5)))


class IGWO_Optimizer:
    def __init__(self, X, y, svm_cfg, n_wolves=30, max_iter=60,
                 penalty_coef=0.01, n_jobs=-1, verbose=True, seed=None,
                 patience=30, search_cv_splits=5, search_cv_repeats=2,
                 n_top_candidates=5):
        self.X = X
        self.y = y
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.n_features = X.shape[1]
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.cache = FitnessCache(X, y, svm_cfg, penalty_coef, self.n_features,
                                   cv_splits=search_cv_splits, cv_repeats=search_cv_repeats)
        # اگه در این تعداد ایتریشن پشت سر هم بهترین فیتنس (آلفا) بهبود پیدا نکنه، بهینه‌سازی زودتر متوقف می‌شه
        self.patience = patience
        # چند تا از بهترین کاندیدهای متمایزی که کل جستجو دیده (نه فقط مسیر alpha) در انتها
        # برگردونده می‌شه تا در لایه‌ی بالاتر با ارزیابی دقیق (نه فیتنس نویزی جستجو) بین‌شون
        # واقعاً بهترین انتخاب بشه — راهکار مقابله با «winner's curse» جستجو روی دیتاست کوچیک
        self.n_top_candidates = n_top_candidates

    def _binarize(self, continuous_pos):
        sig = _sigmoid(continuous_pos)
        rand = self.rng.random(continuous_pos.shape)
        return (rand < sig).astype(int)

    def optimize(self):
        n, d = self.n_wolves, self.n_features
        # مقداردهی اولیه (Eq. 7): هر بعد بین ۰ و ۱
        pos = self.rng.uniform(0, 1, (n, d))
        binary_pos = self._binarize(pos)

        alpha_pos = beta_pos = delta_pos = None
        alpha_bin = None  # بردار باینری واقعی که به alpha_score رسیده (برای خروجی نهایی)
        alpha_score = beta_score = delta_score = -np.inf
        history = []

        # برای early stopping: بهترین فیتنسی که تا الان دیده شده و شمارنده‌ی عدم بهبود
        best_so_far = -np.inf
        no_improve_count = 0

        if self.verbose:
            print("🐺 Starting IGWO (Improved Grey Wolf Optimizer with DLH strategy)...")

        for t in range(self.max_iter):
            # --- فیتنس کل جمعیت فعلی، یک فراخوانی batch موازی ---
            fitness = self.cache.batch_score(binary_pos, n_jobs=self.n_jobs)

            # پیدا کردن سه رهبر: آلفا، بتا، دلتا (بر اساس فیتنس جمعیت همین ایتریشن)
            order = np.argsort(-fitness)
            top3 = order[:3]
            if fitness[top3[0]] > alpha_score:
                alpha_score = fitness[top3[0]]
                alpha_pos = pos[top3[0]].copy()
                alpha_bin = binary_pos[top3[0]].copy()
            if len(top3) > 1 and fitness[top3[1]] > beta_score:
                beta_score, beta_pos = fitness[top3[1]], pos[top3[1]].copy()
            if len(top3) > 2 and fitness[top3[2]] > delta_score:
                delta_score, delta_pos = fitness[top3[2]], pos[top3[2]].copy()
            if alpha_pos is None:
                alpha_pos = pos[order[0]].copy()
                alpha_bin = binary_pos[order[0]].copy()
            if beta_pos is None:
                beta_pos = alpha_pos.copy()
            if delta_pos is None:
                delta_pos = alpha_pos.copy()

            history.append(max(0.0, alpha_score))

            # --- بررسی early stopping (قبل از انجام کار سنگین این ایتریشن) ---
            if alpha_score > best_so_far + 1e-6:
                best_so_far = alpha_score
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.patience:
                if self.verbose:
                    print(f"⏹️  Early stopping: {self.patience} ایتریشن پشت سر هم بدون بهبود "
                          f"(متوقف شد در ایتریشن {t + 1}/{self.max_iter}, بهترین فیتنس={alpha_score:.4f})")
                break

            a = 2 * (1 - t / self.max_iter)  # پارامتر a از ۲ به ۰ کاهش می‌یابد

            # ===================== بخش ۱: آپدیت کلاسیک GWO (وکتورایز، کل جمعیت یک‌جا) =====================
            r1, r2 = self.rng.random((n, d)), self.rng.random((n, d))
            A1, C1 = 2 * a * r1 - a, 2 * r2
            X1 = alpha_pos[None, :] - A1 * np.abs(C1 * alpha_pos[None, :] - pos)

            r1, r2 = self.rng.random((n, d)), self.rng.random((n, d))
            A2, C2 = 2 * a * r1 - a, 2 * r2
            X2 = beta_pos[None, :] - A2 * np.abs(C2 * beta_pos[None, :] - pos)

            r1, r2 = self.rng.random((n, d)), self.rng.random((n, d))
            A3, C3 = 2 * a * r1 - a, 2 * r2
            X3 = delta_pos[None, :] - A3 * np.abs(C3 * delta_pos[None, :] - pos)

            x_gwo = np.clip((X1 + X2 + X3) / 3.0, 0, 1)

            # ===================== بخش ۲: استراتژی DLH (Eq. 8-10، وکتورایز) =====================
            radius = np.linalg.norm(pos - x_gwo, axis=1)  # (n,)

            # فاصله‌ی زوجی بین همه‌ی گرگ‌ها (n x n) بدون حلقه‌ی پایتون
            sq = np.einsum('ij,ij->i', pos, pos)
            dist_sq = np.maximum(sq[:, None] + sq[None, :] - 2 * pos @ pos.T, 0.0)
            dist = np.sqrt(dist_sq)

            # همسایه‌های هر گرگ: کسانی که فاصله‌شون <= radius همون گرگه (خودش همیشه واجد شرایطه چون فاصله‌ش صفره)
            mask = dist <= radius[:, None]
            prio = np.where(mask, self.rng.random((n, n)), -1.0)
            n_idx = np.argmax(prio, axis=1)  # یک همسایه‌ی تصادفی برای هر گرگ

            r_idx = self.rng.integers(0, n, size=n)  # یک گرگ کاملاً تصادفی از کل جمعیت برای هر گرگ

            x_dlh = np.clip(pos + self.rng.random((n, d)) * (pos[n_idx] - pos[r_idx]), 0, 1)

            # ===================== بخش ۳: ارزیابی batch همه‌ی کاندیدها با یک فراخوانی موازی =====================
            cand_pos = np.vstack([x_gwo, x_dlh])          # (2n, d)
            cand_bin = self._binarize(cand_pos)             # (2n, d)
            cand_fit = self.cache.batch_score(cand_bin, n_jobs=self.n_jobs)  # (2n,)

            fit_gwo, fit_dlh = cand_fit[:n], cand_fit[n:]
            dlh_wins = fit_dlh > fit_gwo
            best_cand_pos = np.where(dlh_wins[:, None], x_dlh, x_gwo)
            best_cand_bin = np.where(dlh_wins[:, None], cand_bin[n:], cand_bin[:n])
            best_cand_fit = np.where(dlh_wins, fit_dlh, fit_gwo)

            # فقط اگه کاندید جدید بهتر از موقعیت فعلی گرگ بود جایگزین کن (Eq. 11)
            accept = best_cand_fit > fitness
            pos = np.where(accept[:, None], best_cand_pos, pos)
            binary_pos = np.where(accept[:, None], best_cand_bin, binary_pos)

            if self.verbose and ((t + 1) % 10 == 0 or t == 0):
                print(f"Iteration {t + 1}/{self.max_iter} | Best Fitness: {alpha_score:.4f} "
                      f"| Cache size: {len(self.cache.cache)}")

        final_indices = np.where(alpha_bin == 1)[0]
        if len(final_indices) < 2:
            # fallback: حداقل ۲ ژن با بالاترین وزن پیوسته (خیلی نادر، فقط اگه alpha یه زیرمجموعه‌ی خیلی کوچیک باشه)
            final_indices = np.argsort(-alpha_pos)[:max(2, len(final_indices))]
        selected_genes = self.X.columns[final_indices].tolist()

        # بهترین N کاندید متمایزی که کل جستجو دیده (از کش کامل، نه فقط مسیر alpha)
        sorted_cache = sorted(self.cache.cache.items(), key=lambda kv: kv[1], reverse=True)
        top_candidates = []
        for key, score in sorted_cache:
            if len(key) < 2:
                continue
            top_candidates.append((self.X.columns[list(key)].tolist(), score))
            if len(top_candidates) >= self.n_top_candidates:
                break
        if not top_candidates:
            top_candidates = [(selected_genes, alpha_score)]

        return selected_genes, alpha_score, history, top_candidates
