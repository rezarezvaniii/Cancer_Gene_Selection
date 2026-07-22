"""
IGWO (Improved Grey Wolf Optimizer) — پیاده‌سازی وفادار به مقاله
"Feature selection using game Shapley improved grey wolf optimizer" (Afreen et al., 2025)

نکته‌ی کلیدی نسبت به نسخه‌ی قبلی: اینجا استراتژی DLH (Dimension Learning-based Hunting)
اضافه شده (معادلات ۸ تا ۱۱ مقاله). هر گرگ دو کاندید موقعیت می‌گیره:
  1) X_i-GWO : آپدیت کلاسیک بر اساس alpha/beta/delta
  2) X_i-DLH : یادگیری از همسایه‌های محلی + یک گرگ تصادفی از کل جمعیت
و هرکدوم فیتنس بهتری داشت انتخاب می‌شه. این باعث حفظ تنوع جمعیت و
جلوگیری از همگرایی زودرس می‌شه (نقطه‌ضعف اصلی GWO ساده).
"""

import numpy as np
from .common import FitnessCache


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-10 * (x - 0.5)))


class IGWO_Optimizer:
    def __init__(self, X, y, svm_cfg, n_wolves=30, max_iter=60,
                 penalty_coef=0.01, n_jobs=-1, verbose=True, seed=None,
                 patience=30):
        self.X = X
        self.y = y
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.n_features = X.shape[1]
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.cache = FitnessCache(X, y, svm_cfg, penalty_coef, self.n_features)
        # اگه در این تعداد ایتریشن پشت سر هم بهترین فیتنس (آلفا) بهبود پیدا نکنه، بهینه‌سازی زودتر متوقف می‌شه
        self.patience = patience

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
        alpha_score = beta_score = delta_score = -np.inf
        history = []

        # برای early stopping: بهترین فیتنسی که تا الان دیده شده و شمارنده‌ی عدم بهبود
        best_so_far = -np.inf
        no_improve_count = 0

        if self.verbose:
            print("🐺 Starting IGWO (Improved Grey Wolf Optimizer with DLH strategy)...")

        for t in range(self.max_iter):
            fitness = self.cache.batch_score(binary_pos, n_jobs=self.n_jobs)

            # پیدا کردن سه رهبر: آلفا، بتا، دلتا
            order = np.argsort(-fitness)
            top3 = order[:3]
            if fitness[top3[0]] > alpha_score:
                alpha_score, alpha_pos = fitness[top3[0]], pos[top3[0]].copy()
            if len(top3) > 1 and fitness[top3[1]] > beta_score:
                beta_score, beta_pos = fitness[top3[1]], pos[top3[1]].copy()
            if len(top3) > 2 and fitness[top3[2]] > delta_score:
                delta_score, delta_pos = fitness[top3[2]], pos[top3[2]].copy()
            if alpha_pos is None:
                alpha_pos = pos[order[0]].copy()
            if beta_pos is None:
                beta_pos = alpha_pos.copy()
            if delta_pos is None:
                delta_pos = alpha_pos.copy()

            history.append(max(0.0, alpha_score))

            # --- بررسی early stopping (قبل از انجام کار سنگین DLH برای این ایتریشن) ---
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

            new_pos = np.empty_like(pos)
            for i in range(n):
                # --- بخش ۱: آپدیت کلاسیک GWO (X_i-GWO) ---
                r1, r2 = self.rng.random(d), self.rng.random(d)
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = np.abs(C1 * alpha_pos - pos[i])
                X1 = alpha_pos - A1 * D_alpha

                r1, r2 = self.rng.random(d), self.rng.random(d)
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = np.abs(C2 * beta_pos - pos[i])
                X2 = beta_pos - A2 * D_beta

                r1, r2 = self.rng.random(d), self.rng.random(d)
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = np.abs(C3 * delta_pos - pos[i])
                X3 = delta_pos - A3 * D_delta

                x_gwo = (X1 + X2 + X3) / 3.0
                x_gwo = np.clip(x_gwo, 0, 1)

                # --- بخش ۲: استراتژی DLH (Eq. 8-10) ---
                radius = np.linalg.norm(pos[i] - x_gwo)
                dists = np.linalg.norm(pos - pos[i], axis=1)
                neighbor_idx = np.where(dists <= radius)[0]
                if len(neighbor_idx) == 0:
                    neighbor_idx = np.array([i])
                n_idx = self.rng.choice(neighbor_idx)
                r_idx = self.rng.integers(0, n)
                x_dlh = pos[i] + self.rng.random(d) * (pos[n_idx] - pos[r_idx])
                x_dlh = np.clip(x_dlh, 0, 1)

                # --- بخش ۳: انتخاب بهترین کاندید (Eq. 11) ---
                cand = np.stack([x_gwo, x_dlh])
                cand_bin = self._binarize(cand)
                cand_fit = self.cache.batch_score(cand_bin, n_jobs=1)
                best_cand = cand[np.argmax(cand_fit)]
                best_fit = cand_fit.max()

                # فقط اگه بهتر از موقعیت فعلی بود جایگزین کن
                if best_fit > fitness[i]:
                    new_pos[i] = best_cand
                else:
                    new_pos[i] = pos[i]

            pos = new_pos
            binary_pos = self._binarize(pos)

            if self.verbose and ((t + 1) % 10 == 0 or t == 0):
                print(f"Iteration {t + 1}/{self.max_iter} | Best Fitness: {alpha_score:.4f} "
                      f"| Cache size: {len(self.cache.cache)}")

        final_binary = self._binarize(alpha_pos)
        final_indices = np.where(final_binary == 1)[0]
        if len(final_indices) < 2:
            # fallback: حداقل ۲ ژن با بالاترین وزن پیوسته
            final_indices = np.argsort(-alpha_pos)[:max(2, len(final_indices))]
        selected_genes = self.X.columns[final_indices].tolist()

        return selected_genes, alpha_score, history