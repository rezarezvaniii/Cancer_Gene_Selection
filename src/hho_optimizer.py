import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
import warnings

warnings.filterwarnings('ignore')

class HHO_Optimizer:
    def __init__(self, X, y, n_hawks=30, max_iter=100):
        self.X = X
        self.y = y
        self.n_hawks = n_hawks
        self.max_iter = max_iter
        self.n_features = X.shape[1]
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def fitness_function(self, binary_pos):
        selected_indices = np.where(binary_pos == 1)[0]
        if len(selected_indices) < 2: return 0
        
        X_subset = self.X.iloc[:, selected_indices]
        clf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
        
        try:
            scores = cross_val_score(clf, X_subset, self.y, cv=self.cv, n_jobs=-1)
            return scores.mean() - (0.01 * len(selected_indices) / self.n_features)
        except Exception as e:
            # اگه بازم صفر شد، این مسیج بهت می‌گه چرا
            # print(f"Error in CV: {e}") 
            return 0

    def optimize(self):
        # مقداردهی اولیه شاهین‌ها (موقعیت بین 0 و 1)
        pos = np.random.uniform(0, 1, (self.n_hawks, self.n_features))
        rabbit_pos = np.zeros(self.n_features)
        rabbit_score = -np.inf
        
        history = []
        print(f"🚀 Starting Harris Hawks Optimization (HHO)...")

        for t in range(self.max_iter):
            # مرحله ۱: ارزیابی فیتنس تمام شاهین‌ها
            for i in range(self.n_hawks):
                # تبدیل موقعیت پیوسته به باینری (۰ یا ۱)
                binary_pos = np.where(pos[i, :] > 0.5, 1, 0)
                fitness = self.fitness_function(binary_pos)
                
                # بروزرسانی بهترین موقعیت پیدا شده (خرگوش)
                if fitness > rabbit_score:
                    rabbit_score = fitness
                    rabbit_pos = pos[i, :].copy()

            history.append(max(0, rabbit_score))
            
            # مرحله ۲: آپدیت پارامترهای HHO (انرژی فرار)
            E0 = 2 * np.random.random() - 1  # انرژی اولیه تصادفی
            E = 2 * E0 * (1 - (t / self.max_iter))  # کاهش انرژی در طول زمان

            for i in range(self.n_hawks):
                if abs(E) >= 1:  # فاز اکتشاف (Exploration)
                    q = np.random.random()
                    rand_idx = np.random.randint(0, self.n_hawks)
                    if q >= 0.5: # پرواز تصادفی شاهین‌ها
                        pos[i, :] = pos[rand_idx, :] - np.random.random() * abs(pos[rand_idx, :] - 2 * np.random.random() * pos[i, :])
                    else: # نشستن روی درخت‌های مختلف
                        pos[i, :] = (rabbit_pos - pos.mean(axis=0)) - np.random.random() * np.random.random()
                
                else:  # فاز بهره‌برداری (Exploitation)
                    r = np.random.random()
                    if r >= 0.5 and abs(E) >= 0.5: # محاصره نرم (Soft Besiege)
                        jump_strength = 2 * (1 - np.random.random())
                        pos[i, :] = (rabbit_pos - pos[i, :]) - E * abs(jump_strength * rabbit_pos - pos[i, :])
                    elif r >= 0.5 and abs(E) < 0.5: # محاصره سخت (Hard Besiege)
                        pos[i, :] = rabbit_pos - E * abs(rabbit_pos - pos[i, :])
                    else: # حملات متوالی و زیگزاگی (Zigzag attack)
                        pos[i, :] = rabbit_pos + E * np.random.uniform(-1, 1, self.n_features)

            # محدود کردن مقادیر در بازه 0 و 1
            pos = np.clip(pos, 0, 1)

            # چاپ پیشرفت هر ۱۰ تکرار
            if (t+1) % 10 == 0 or t == 0:
                print(f"Iteration {t+1}/{self.max_iter} | Best Fitness: {rabbit_score:.4f}")

        # استخراج ژن‌های نهایی از بهترین شاهین (Rabbit)
        final_binary = np.where(rabbit_pos > 0.5, 1, 0)
        final_indices = np.where(final_binary == 1)[0]
        selected_genes = self.X.columns[final_indices].tolist()
        
        return selected_genes, rabbit_score, history