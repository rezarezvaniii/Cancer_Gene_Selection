import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
import warnings

warnings.filterwarnings('ignore')

class HHO_Optimizer:
    def __init__(self, X, y, n_hawks=30, max_iter=80):
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
        # استفاده از SVC با کرنل RBF برای جلوگیری از دقت کاذب 100%
        clf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
        
        try:
            # استفاده از تمام توان CPU شما
            scores = cross_val_score(clf, X_subset, self.y, cv=self.cv, n_jobs=-1)
            accuracy = scores.mean()
            # جریمه سنگین‌تر برای تعداد ویژگی‌ها (0.01) جهت واقعی‌تر شدن نتایج
            penalty = 0.01 * (len(selected_indices) / self.n_features)
            return accuracy - penalty
        except:
            return 0

    def optimize(self):
        # مقداردهی اولیه شاهین‌ها (موقعیت پیوسته)
        pos = np.random.uniform(0, 1, (self.n_hawks, self.n_features))
        rabbit_pos = np.zeros(self.n_features)
        rabbit_score = -np.inf
        history = []

        print(f"Starting Harris Hawks Optimization (HHO)...")

        for t in range(self.max_iter):
            for i in range(self.n_hawks):
                # تبدیل به باینری با آستانه 0.5
                binary_pos = np.where(pos[i, :] > 0.5, 1, 0)
                fitness = self.fitness_function(binary_pos)
                
                if fitness > rabbit_score:
                    rabbit_score, rabbit_pos = fitness, pos[i, :].copy()

            history.append(max(0, rabbit_score))
            
            # انرژی فرار شکار (خرگوش)
            E0 = 2 * np.random.random() - 1
            E = 2 * E0 * (1 - (t / self.max_iter))

            for i in range(self.n_hawks):
                if abs(E) >= 1:  # فاز اکتشاف (Exploration)
                    q = np.random.random()
                    if q >= 0.5:
                        pos[i, :] = pos[np.random.randint(self.n_hawks), :] - np.random.random() * abs(pos[np.random.randint(self.n_hawks), :] - 2 * np.random.random() * pos[i, :])
                    else:
                        pos[i, :] = (rabbit_pos - pos.mean(axis=0)) - np.random.random() * np.random.random()
                
                else:  # فاز بهره‌برداری (Exploitation)
                    r = np.random.random()
                    if r >= 0.5 and abs(E) >= 0.5: # محاصره نرم
                        jump_strength = 2 * (1 - np.random.random())
                        pos[i, :] = (rabbit_pos - pos[i, :]) - E * abs(jump_strength * rabbit_pos - pos[i, :])
                    elif r >= 0.5 and abs(E) < 0.5: # محاصره سخت
                        pos[i, :] = rabbit_pos - E * abs(rabbit_pos - pos[i, :])
                    else: # حملات متوالی
                        pos[i, :] = rabbit_pos + E * np.random.uniform(-1, 1, self.n_features)

            pos = np.clip(pos, 0, 1)
            if (t+1) % 10 == 0 or t == 0:
                print(f"Iteration {t+1}/{self.max_iter} | Best Fitness: {rabbit_score:.4f}")

        final_indices = np.where(rabbit_pos > 0.5)[0]
        return self.X.columns[final_indices].tolist(), rabbit_score, history