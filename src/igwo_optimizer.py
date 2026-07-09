import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import LinearSVC
import warnings

warnings.filterwarnings('ignore')

class IGWO_Optimizer:
    def __init__(self, X, y, n_wolves=100, max_iter=100):
        self.X = X
        self.y = y
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.n_features = X.shape[1]
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def fitness_function(self, binary_pos):
        selected_indices = np.where(binary_pos == 1)[0]
        # اگر هیچ ژنی انتخاب نشود یا فقط یکی باشد، فیتنس صفر برمی‌گردانیم
        if len(selected_indices) < 2: 
            return 0
        
        X_subset = self.X.iloc[:, selected_indices]
        # استفاده از LinearSVC مطابق با بهترین متدهای انتخاب ژن در داده‌های سرطان
        clf = LinearSVC(C=1.0, max_iter=5000, random_state=42)
        
        try:
            scores = cross_val_score(clf, X_subset, self.y, cv=self.cv)
            accuracy = scores.mean()
            # جریمه بسیار کوچک برای تشویق به انتخاب ژن‌های کمتر (طبق مقاله)
            penalty = 0.001 * (len(selected_indices) / self.n_features)
            return accuracy - penalty
        except:
            return 0

    def optimize(self):
        # مقداردهی اولیه: هر گرگ با احتمال ۵۰٪ هر ژن را انتخاب می‌کند
        binary_positions = np.random.randint(0, 2, (self.n_wolves, self.n_features))
        continuous_positions = binary_positions.astype(float)
        
        alpha_pos, alpha_score = None, -np.inf
        beta_pos, beta_score = None, -np.inf
        delta_pos, delta_score = None, -np.inf
        
        history = []

        print(f"Starting Elite IGWO (Goal: >95% Accuracy)...")
        
        for t in range(self.max_iter):
            # فاز ۱: ارزیابی فیتنس و پیدا کردن رهبران (Alpha, Beta, Delta)
            for i in range(self.n_wolves):
                fitness = self.fitness_function(binary_positions[i, :])
                
                if fitness > alpha_score:
                    alpha_score, alpha_pos = fitness, continuous_positions[i, :].copy()
                elif fitness > beta_score:
                    beta_score, beta_pos = fitness, continuous_positions[i, :].copy()
                elif fitness > delta_score:
                    delta_score, delta_pos = fitness, continuous_positions[i, :].copy()

            history.append(max(0, alpha_score))
            
            # پارامتر کنترلی a که از ۲ به ۰ کاهش می‌یابد
            a = 2 * (1 - (t / self.max_iter)) 

            # فاز ۲: آپدیت موقعیت تمام گرگ‌ها
            for i in range(self.n_wolves):
                for j in range(self.n_features):
                    # فرمول‌های ریاضی حرکت گرگ‌ها (GWO)
                    # رهبر اول: Alpha
                    r1, r2 = np.random.random(), np.random.random()
                    A1, C1 = 2*a*r1 - a, 2*r2
                    D_alpha = abs(C1 * alpha_pos[j] - continuous_positions[i, j])
                    X1 = alpha_pos[j] - A1 * D_alpha
                    
                    # رهبر دوم: Beta
                    r1, r2 = np.random.random(), np.random.random()
                    A2, C2 = 2*a*r1 - a, 2*r2
                    D_beta = abs(C2 * beta_pos[j] - continuous_positions[i, j])
                    X2 = beta_pos[j] - A2 * D_beta
                    
                    # رهبر سوم: Delta
                    r1, r2 = np.random.random(), np.random.random()
                    A3, C3 = 2*a*r1 - a, 2*r2
                    D_delta = abs(C3 * delta_pos[j] - continuous_positions[i, j])
                    X3 = delta_pos[j] - A3 * D_delta
                    
                    # میانگین‌گیری برای موقعیت پیوسته جدید
                    continuous_positions[i, j] = (X1 + X2 + X3) / 3
                    
                    # تبدیل به باینری با تابع انتقال سیگموئید (S-shaped)
                    # این بخش باعث انتخاب هوشمند ژن‌ها می‌شود
                    sigmoid = 1 / (1 + np.exp(-10 * (continuous_positions[i, j] - 0.5)))
                    if np.random.random() < sigmoid:
                        binary_positions[i, j] = 1
                    else:
                        binary_positions[i, j] = 0

            # چاپ وضعیت هر ۱۰ تکرار یکبار (خارج از حلقه گرگ‌ها)
            if (t+1) % 10 == 0 or t == 0:
                print(f"Iteration {t+1}/{self.max_iter} | Best Fitness: {alpha_score:.4f}")

        # استخراج نهایی نام ژن‌های انتخاب شده توسط رهبر گروه (Alpha)
        final_indices = np.where(alpha_pos > 0.5)[0]
        selected_genes = self.X.columns[final_indices].tolist()
        
        return selected_genes, alpha_score, history