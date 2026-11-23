"""
ISBOA (Improved Secretary Bird Optimization Algorithm) — همون اسکلت IGWO (kSV-IGWO,
Afreen et al. 2025)، با این تفاوت که هسته‌ی «GWO» با هسته‌ی الگوریتم SBOA جایگزین شده.
استراتژی DLH (ایده‌ی بهبودی که روی IGWO خوب جواب می‌داد) عیناً از igwo_optimizer.py کپی
شده و فقط ورودیش (x_base) حالا از SBOA میاد، نه از GWO.

مرجع SBOA:
Fu, Y., Liu, D., Chen, J., et al. (2024).
"Secretary bird optimization algorithm: a new metaheuristic for solving global optimization
problems." Artificial Intelligence Review, 57, 123. (Springer)
DOI: 10.1007/s10462-024-10729-y

معادلات هسته‌ی SBOA (بخش ۱، جایگزین GWO) — سه مرحله‌ی اکتشاف بر اساس نسبت ایتریشن t/T
(همون بخشی که به‌عنوان جایگزین GWO انتخاب شد؛ استراتژی‌های exploitation مقاله اینجا عمداً
اضافه نشدن چون نقش «بهره‌برداری» رو در این اسکلت، DLH (بخش ۲) ایفا می‌کنه):
  t/T < 1/3   (جست‌وجوی مار):       x_new = x_i + (x_r1 - x_r2)·rand
  1/3<=t/T<2/3 (تعقیب مار):         x_new = best + exp(-4t/T)·(R_B-0.5)·(best - x_i)   [R_B~N(0,1)]
  t/T >= 2/3   (حمله‌ی نهایی):       x_new = best + (1-t/T)²·(t/T)·x_i·R_L              [R_L = پرواز لِوی]
"""

import numpy as np
from .common import FitnessCache, levy_flight


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-10 * (x - 0.5)))


class ISBOA_Optimizer:
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
        self.patience = patience
        self.n_top_candidates = n_top_candidates

    def _binarize(self, continuous_pos):
        sig = _sigmoid(continuous_pos)
        rand = self.rng.random(continuous_pos.shape)
        return (rand < sig).astype(int)

    def optimize(self):
        n, d = self.n_wolves, self.n_features
        pos = self.rng.uniform(0, 1, (n, d))
        binary_pos = self._binarize(pos)

        alpha_pos = None
        alpha_bin = None
        alpha_score = -np.inf
        history = []

        best_so_far = -np.inf
        no_improve_count = 0

        if self.verbose:
            print("🦅 Starting ISBOA (Improved Secretary Bird Optimizer with DLH strategy)...")

        for t in range(self.max_iter):
            fitness = self.cache.batch_score(binary_pos, n_jobs=self.n_jobs)

            order = np.argsort(-fitness)
            if fitness[order[0]] > alpha_score:
                alpha_score = fitness[order[0]]
                alpha_pos = pos[order[0]].copy()
                alpha_bin = binary_pos[order[0]].copy()

            history.append(max(0.0, alpha_score))

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

            # ===================== بخش ۱: هسته‌ی SBOA (وکتورایز، کل جمعیت یک‌جا) =====================
            T = self.max_iter
            frac = (t + 1) / T

            if frac < 1 / 3:
                r1_idx = self.rng.integers(0, n, size=n)
                r2_idx = self.rng.integers(0, n, size=n)
                x_base = pos + (pos[r1_idx] - pos[r2_idx]) * self.rng.random((n, d))
            elif frac < 2 / 3:
                RB = self.rng.normal(0, 1, (n, d))
                x_base = alpha_pos[None, :] + np.exp(-4 * frac) * (RB - 0.5) * (alpha_pos[None, :] - pos)
            else:
                RL = levy_flight((n, d), self.rng)
                x_base = alpha_pos[None, :] + ((1 - frac) ** 2) * frac * pos * RL

            x_base = np.clip(x_base, 0, 1)

            # ===================== بخش ۲: استراتژی DLH (Eq. 8-10 مقاله‌ی IGWO، وکتورایز) =====================
            radius = np.linalg.norm(pos - x_base, axis=1)

            sq = np.einsum('ij,ij->i', pos, pos)
            dist_sq = np.maximum(sq[:, None] + sq[None, :] - 2 * pos @ pos.T, 0.0)
            dist = np.sqrt(dist_sq)

            mask = dist <= radius[:, None]
            prio = np.where(mask, self.rng.random((n, n)), -1.0)
            n_idx = np.argmax(prio, axis=1)

            r_idx = self.rng.integers(0, n, size=n)

            x_dlh = np.clip(pos + self.rng.random((n, d)) * (pos[n_idx] - pos[r_idx]), 0, 1)

            # ===================== بخش ۳: ارزیابی batch همه‌ی کاندیدها با یک فراخوانی موازی =====================
            cand_pos = np.vstack([x_base, x_dlh])
            cand_bin = self._binarize(cand_pos)
            cand_fit = self.cache.batch_score(cand_bin, n_jobs=self.n_jobs)

            fit_base, fit_dlh = cand_fit[:n], cand_fit[n:]
            dlh_wins = fit_dlh > fit_base
            best_cand_pos = np.where(dlh_wins[:, None], x_dlh, x_base)
            best_cand_bin = np.where(dlh_wins[:, None], cand_bin[n:], cand_bin[:n])
            best_cand_fit = np.where(dlh_wins, fit_dlh, fit_base)

            accept = best_cand_fit > fitness
            pos = np.where(accept[:, None], best_cand_pos, pos)
            binary_pos = np.where(accept[:, None], best_cand_bin, binary_pos)

            if self.verbose and ((t + 1) % 10 == 0 or t == 0):
                print(f"Iteration {t + 1}/{self.max_iter} | Best Fitness: {alpha_score:.4f} "
                      f"| Cache size: {len(self.cache.cache)}")

        final_indices = np.where(alpha_bin == 1)[0]
        if len(final_indices) < 2:
            final_indices = np.argsort(-alpha_pos)[:max(2, len(final_indices))]
        selected_genes = self.X.columns[final_indices].tolist()

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
