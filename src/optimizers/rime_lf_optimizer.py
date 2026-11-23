"""
RIME-LF — هسته‌ی الگوریتم RIME (Su et al. 2023، Neurocomputing) ترکیب‌شده با
Lévy Flight به‌عنوان استراتژی بهبود.

مرجع RIME (فرمول‌ها از سورس رسمی نویسنده‌ها گرفته شده، نه از حافظه):
Su, H., Zhao, D., Heidari, A.A., Liu, L., Zhang, X., Mafarja, M., Chen, H. (2023).
"RIME: A physics-based optimization." Neurocomputing, 532, 183-214. (Elsevier)
DOI: 10.1016/j.neucom.2023.02.010
سورس رسمی: https://github.com/aliasgharheidaricom/RIME-A-physics-based-optimization

معادلات هسته‌ی RIME (بخش ۱):
  E = sqrt(t/T)                                                  — پیشرفت جستجو
  W = 5
  RimeFactor = (rand-0.5)*2*cos(pi*t/(T/10))*(1-round(t*W/T)/W)  — دامنه‌ی نوسان کاهشی
  Soft-rime (اکتشاف، هر بعد با احتمال rand<E):
      x_new = best + RimeFactor * rand(0,1)   [چون فضای جستجوی ما همیشه [0,1] است]
  Hard-rime puncture (بهره‌برداری، هر بعد با احتمال rand < normalized_rate(i)):
      x_new = best

⚠️ تطبیق لازم برای ماکسیمایز: RIME اصلی برای مینیمم‌سازی نوشته شده (Rime_rates=فیتنس
خام، هرچی رتبه/فیتنس عامل بدتر بود -> normalized_rate بزرگ‌تر -> احتمال بیشتر hard-rime
به‌سمت بهترین). چون فیتنس ما "هرچی بزرگ‌تر بهتر" است، هزینه رو به‌صورت
cost_i = max(fitness) - fitness_i تعریف می‌کنیم (برای بهترین عامل صفر، برای بدترین
بیشترین) و normalized_rate را از روی این cost می‌سازیم — دقیقاً همون رفتار کیفی نسخه‌ی
اصلی: عامل‌های ضعیف‌تر احتمال بیشتری برای پرش مستقیم به‌سمت بهترین دارن.

مرجع Lévy Flight (بخش ۲):
Yang, X.S., Deb, S. (2009). "Cuckoo search via Lévy flights." 2009 World Congress
on Nature & Biologically Inspired Computing (NaBIC), 210-214. IEEE.
از تابع مشترک levy_flight (الگوریتم Mantegna، src/optimizers/common.py) استفاده
می‌کنه — همون تابعی که isboa_optimizer.py هم استفاده کرده. برای هر عامل، بعد از
محاسبه‌ی موقعیت جدید طبق هسته‌ی RIME (x_base)، یه جهش لِوی در راستای فاصله‌ی عامل
تا بهترین جواب فعلی (alpha_pos) اضافه می‌شه:
  x_levy = clip(x_base + step_size * levy_flight(...) * (alpha_pos - x_base), 0, 1)
step_size با پیشرفت جستجو (frac=(t+1)/T) به‌صورت خطی از ۱ به ۰ کاهش می‌یابد — هماهنگ
با سبک ضریب‌های زمان‌محور خود هسته‌ی RIME (E و RimeFactor بالا): اوایل جستجو جهش‌های
بزرگ‌تر (اکتشاف)، اواخر جستجو جهش‌های ظریف‌تر (بهره‌برداری ریزتر). سپس x_base و x_levy
هر دو باینری و فیتنس‌شون با یک فراخوانی batch سنجیده می‌شه؛ برای هر عامل فیتترین
کاندید نگه داشته می‌شه و فقط اگه از موقعیت فعلی عامل بهتر بود (greedy replacement)
جایگزین می‌شه.
"""

import numpy as np
from .common import FitnessCache, levy_flight


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-10 * (x - 0.5)))


class RIMELF_Optimizer:
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
            print("🧊 Starting RIME-LF (RIME with Lévy Flight)...")

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

            # ===================== بخش ۱: هسته‌ی RIME (وکتورایز، کل جمعیت یک‌جا) =====================
            T = self.max_iter
            E = np.sqrt((t + 1) / T)
            W = 5
            RimeFactor = (self.rng.random() - 0.5) * 2 * np.cos(np.pi * (t + 1) / (T / 10)) * \
                         (1 - round((t + 1) * W / T) / W)

            x_base = pos.copy()

            # Soft-rime (اکتشاف)
            soft_mask = self.rng.random((n, d)) < E
            x_soft = alpha_pos[None, :] + RimeFactor * self.rng.random((n, d))
            x_base = np.where(soft_mask, x_soft, x_base)

            # Hard-rime puncture (بهره‌برداری) — نرخ بر اساس هزینه‌ی نرمالایز‌شده (تطبیق ماکسیمایز، بالا توضیح داده شد)
            cost = fitness.max() - fitness + 1e-12
            normalized_rate = cost / (np.linalg.norm(cost) + 1e-12)
            hard_mask = self.rng.random((n, d)) < normalized_rate[:, None]
            x_base = np.where(hard_mask, alpha_pos[None, :], x_base)

            x_base = np.clip(x_base, 0, 1)

            # ===================== بخش ۲: Lévy Flight =====================
            # جهش لِوی در راستای فاصله‌ی هر عامل تا بهترین جواب فعلی (alpha_pos)؛ step_size
            # با پیشرفت جستجو خطی از ۱ به ۰ کاهش می‌یابد (جهش‌های بزرگ‌تر اوایل کار، ظریف‌تر
            # در پایان) — هماهنگ با سبک ضریب‌های زمان‌محور هسته‌ی RIME بالا (E, RimeFactor).
            frac = (t + 1) / T
            step_size = 1 - frac
            RL = levy_flight((n, d), self.rng)
            x_levy = np.clip(x_base + step_size * RL * (alpha_pos[None, :] - x_base), 0, 1)

            # ===================== بخش ۳: ارزیابی batch همه‌ی کاندیدها با یک فراخوانی موازی =====================
            cand_pos = np.vstack([x_base, x_levy])
            cand_bin = self._binarize(cand_pos)
            cand_fit = self.cache.batch_score(cand_bin, n_jobs=self.n_jobs)

            fit_base, fit_levy = cand_fit[:n], cand_fit[n:]
            levy_wins = fit_levy > fit_base
            best_cand_pos = np.where(levy_wins[:, None], x_levy, x_base)
            best_cand_bin = np.where(levy_wins[:, None], cand_bin[n:], cand_bin[:n])
            best_cand_fit = np.where(levy_wins, fit_levy, fit_base)

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
