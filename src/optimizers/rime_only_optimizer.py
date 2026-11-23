"""
RIME_Only — نسخه‌ی خام الگوریتم RIME، بدون Lévy Flight. این فقط برای Ablation
Study ساخته شده (که اثر جداگانه‌ی Lévy Flight رو نشون بده) — عیناً همون
rime_lf_optimizer.py است، فقط بخش «Lévy Flight» (که در آنجا بخش ۲ بود) حذف
شده و x_base مستقیماً به‌عنوان کاندید نهایی هر ایتریشن استفاده می‌شه.

مرجع RIME: Su, H., Zhao, D., Heidari, A.A., Liu, L., Zhang, X., Mafarja, M.,
Chen, H. (2023). "RIME: A physics-based optimization." Neurocomputing, 532,
183-214.
"""

import numpy as np
from .common import FitnessCache


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-10 * (x - 0.5)))


class RIMEOnly_Optimizer:
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
            print("🧊 Starting RIME-Only (RIME core, no Lévy Flight — ablation variant)...")

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
                    print(f"⏹️  Early stopping at iteration {t + 1}/{self.max_iter}, best fitness={alpha_score:.4f}")
                break

            # ===================== RIME core only (identical to rime_lf_optimizer.py) =====================
            T = self.max_iter
            E = np.sqrt((t + 1) / T)
            W = 5
            RimeFactor = (self.rng.random() - 0.5) * 2 * np.cos(np.pi * (t + 1) / (T / 10)) * \
                         (1 - round((t + 1) * W / T) / W)

            x_base = pos.copy()

            soft_mask = self.rng.random((n, d)) < E
            x_soft = alpha_pos[None, :] + RimeFactor * self.rng.random((n, d))
            x_base = np.where(soft_mask, x_soft, x_base)

            cost = fitness.max() - fitness + 1e-12
            normalized_rate = cost / (np.linalg.norm(cost) + 1e-12)
            hard_mask = self.rng.random((n, d)) < normalized_rate[:, None]
            x_base = np.where(hard_mask, alpha_pos[None, :], x_base)

            x_base = np.clip(x_base, 0, 1)

            # ⚠️ تفاوت با RIME-LF: اینجا دیگه Lévy Flight اعمال نمی‌شه — x_base مستقیماً
            # کاندید نهایی این ایتریشن است.
            cand_bin = self._binarize(x_base)
            cand_fit = self.cache.batch_score(cand_bin, n_jobs=self.n_jobs)

            accept = cand_fit > fitness
            pos = np.where(accept[:, None], x_base, pos)
            binary_pos = np.where(accept[:, None], cand_bin, binary_pos)

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
