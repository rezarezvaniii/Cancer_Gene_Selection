"""
IGO (Improved Growth Optimizer) — همون اسکلت IGWO (kSV-IGWO, Afreen et al. 2025)، با این
تفاوت که هسته‌ی «GWO» با هسته‌ی الگوریتم Growth Optimizer (GO) جایگزین شده. استراتژی DLH
(ایده‌ی بهبودی که روی IGWO خوب جواب می‌داد) عیناً از igwo_optimizer.py کپی شده و فقط
ورودیش (x_base) حالا از GO میاد، نه از GWO.

مرجع GO (فرمول‌ها و پارامترهای ثابت از سورس رسمی نویسنده‌ها گرفته شده، نه از حافظه):
Zhang, Q., Gao, H., Zhan, Z.-H., Li, J., Zhang, H. (2023).
"Growth Optimizer: A powerful metaheuristic algorithm for solving continuous and discrete
global optimization problems." Knowledge-Based Systems, 261, 110206. (Elsevier)
DOI: 10.1016/j.knosys.2022.110206
سورس رسمی: https://github.com/tsingke/Growth-Optimizer  (P1=5, P2=0.001, P3=0.3 — ثابت‌های
خود نویسنده‌ها؛ P2 در کد اصلی مربوط به بخش دیگری‌ست که اینجا استفاده نشده).

معادلات هسته‌ی GO (بخش ۱، جایگزین GWO):
  Learning phase: هر عامل i بر اساس فاصله‌اش تا ۴ نقطه‌ی مرجع (بهترین فعلی، یک عامل «بهتر»
  تصادفی از میان P1 نفر برتر، یک عامل «بدتر» تصادفی از میان P1 نفر ته جدول، و دو عامل کاملاً
  تصادفی) و با وزن‌دهی نسبت‌فاصله (LF) × «ضریب مقیاس» (SF) جابه‌جا می‌شه:
      Gap1=Best-Better, Gap2=Best-Worst, Gap3=Better-Worst, Gap4=Rand1-Rand2
      LF_k = Distance_k / ΣDistance ,  x_new = x_i + Σ LF_k·SF·Gap_k
  Reflection phase: روی خروجی Learning، هر بعد با احتمال P3 به‌سمت یک عامل تصادفی از P1 نفر
  برتر کشیده می‌شه، و با احتمال adaptive کوچک (AF) کاملاً رندوم بازتولید می‌شه.

⚠️ دو تطبیق آگاهانه نسبت به مقاله‌ی اصلی (چون معماری اسکلت IGWO فقط جای یک "x_base" قبل
از DLH داره، نه دو مرحله‌ی جدا با accept-if-better مستقل):
  1) Learning و Reflection زنجیره‌ای اجرا می‌شن (خروجی Learning ورودی Reflection می‌شه) و
     فقط نتیجه‌ی نهایی (x_base) به DLH داده می‌شه — نه این‌که هرکدوم جدا greedy-accept بشن.
  2) SF (ضریب مقیاس) و "بهترین/بدترین" در مقاله‌ی اصلی برای مینیمم‌سازی تعریف شدن (فیتنس
     پایین‌تر = بهتر). چون فیتنس ما برعکسه (بالاتر = بهتر)، از cost_i = max(fitness)-fitness_i
     به‌جای فیتنس خام استفاده می‌شه — همون رفتار کیفی (SF بزرگ برای عامل‌های ضعیف، کوچک
     برای عامل‌های قوی) حفظ می‌شه.
"""

import numpy as np
from .common import FitnessCache


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-10 * (x - 0.5)))


class IGO_Optimizer:
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
        # ثابت‌های خود مقاله‌ی GO (سورس رسمی نویسنده‌ها)
        self.P1 = min(5, max(2, n_wolves // 2))
        self.P3 = 0.3

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
            print("🌱 Starting IGO (Improved Growth Optimizer with DLH strategy)...")

        for t in range(self.max_iter):
            fitness = self.cache.batch_score(binary_pos, n_jobs=self.n_jobs)

            order = np.argsort(-fitness)  # order[0] = بهترین عامل این ایتریشن
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

            # ===================== بخش ۱: هسته‌ی GO (وکتورایز، کل جمعیت یک‌جا) =====================
            P1 = self.P1
            cost = fitness.max() - fitness + 1e-12         # هرچی کمتر بهتر (تطبیق ماکسیمایز)
            SF = cost / (cost.max() + 1e-12)                # (n,)

            best_x = pos[order[0]]
            better_idx = order[self.rng.integers(1, P1, size=n)]
            worst_idx = order[self.rng.integers(n - P1, n, size=n)]
            l1_idx = self.rng.integers(0, n, size=n)
            l2_idx = self.rng.integers(0, n, size=n)

            better_x, worst_x = pos[better_idx], pos[worst_idx]
            l1_x, l2_x = pos[l1_idx], pos[l2_idx]

            Gap1 = best_x[None, :] - better_x
            Gap2 = best_x[None, :] - worst_x
            Gap3 = better_x - worst_x
            Gap4 = l1_x - l2_x

            Dist1, Dist2 = np.linalg.norm(Gap1, axis=1), np.linalg.norm(Gap2, axis=1)
            Dist3, Dist4 = np.linalg.norm(Gap3, axis=1), np.linalg.norm(Gap4, axis=1)
            SumDist = Dist1 + Dist2 + Dist3 + Dist4 + 1e-12
            LF1, LF2 = Dist1 / SumDist, Dist2 / SumDist
            LF3, LF4 = Dist3 / SumDist, Dist4 / SumDist

            scale = SF[:, None]
            x_learn = pos + (LF1[:, None] * Gap1 + LF2[:, None] * Gap2 +
                              LF3[:, None] * Gap3 + LF4[:, None] * Gap4) * scale

            # Reflection phase (روی خروجی Learning)
            reflect_mask = self.rng.random((n, d)) < self.P3
            R = pos[order[self.rng.integers(0, P1, size=n)]]
            x_reflect = x_learn + (R - x_learn) * self.rng.random((n, d))
            x_base = np.where(reflect_mask, x_reflect, x_learn)

            AF = 0.01 + (0.1 - 0.01) * (1 - (t + 1) / self.max_iter)
            reinit_mask = reflect_mask & (self.rng.random((n, d)) < AF)
            x_random = self.rng.uniform(0, 1, (n, d))
            x_base = np.where(reinit_mask, x_random, x_base)

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
