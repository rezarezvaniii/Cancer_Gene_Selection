import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import LinearSVC
import warnings

warnings.filterwarnings('ignore')

class IGWO_Optimizer:
    def __init__(self, X, y, n_wolves=40, max_iter=80):
        self.X = X
        self.y = y
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.n_features = X.shape[1]
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def fitness_function(self, position):
        # تبدیل هوشمند به باینری
        selected_indices = np.where(position > 0.5)[0]
        if len(selected_indices) == 0: return 0
        
        X_subset = self.X.iloc[:, selected_indices]
        # استفاده از LinearSVC که در ابعاد بالا بسیار سریعتر و دقیق تر از SVC معمولی است
        clf = LinearSVC(C=0.5, class_weight='balanced', max_iter=2000, random_state=42)
        
        try:
            scores = cross_val_score(clf, X_subset, self.y, cv=self.cv)
            accuracy = scores.mean()
            # جریمه ویژگی برای رسیدن به ژن های کمتر (معیار طلایی)
            penalty = 0.005 * (len(selected_indices) / self.n_features)
            return accuracy - penalty
        except:
            return 0

    def optimize(self):
        # مقداردهی اولیه هوشمند
        positions = np.random.uniform(0, 1, (self.n_wolves, self.n_features))
        alpha_pos, alpha_score = None, -np.inf
        beta_pos, beta_score = None, -np.inf
        delta_pos, delta_score = None, -np.inf
        
        history = []
        stall_counter = 0 # شمارنده برای تشخیص درجا زدن

        print(f"Starting Elite IGWO (Goal: 100% Accuracy)...")
        
        for t in range(self.max_iter):
            last_best = alpha_score
            
            for i in range(self.n_wolves):
                fitness = self.fitness_function(positions[i, :])
                
                if fitness > alpha_score:
                    alpha_score, alpha_pos = fitness, positions[i, :].copy()
                    stall_counter = 0
                elif fitness > beta_score:
                    beta_score, beta_pos = fitness, positions[i, :].copy()
                elif fitness > delta_score:
                    delta_score, delta_pos = fitness, positions[i, :].copy()

            history.append(max(0, alpha_score))
            
            # استراتژی فرار از تله (اگر 7 بار درجا زد، جهش ایجاد کن)
            if alpha_score <= last_best: stall_counter += 1
            
            a = 2 * (1 - (t / self.max_iter)) # پارامتر کنترلی خطی

            for i in range(self.n_wolves):
                # حرکت GWO
                r1, r2 = np.random.random(), np.random.random()
                A, C = 2*a*r1 - a, 2*r2
                X1 = alpha_pos - A * abs(C * alpha_pos - positions[i, :])
                
                X2 = beta_pos - A * abs(C * beta_pos - positions[i, :])
                X3 = delta_pos - A * abs(C * delta_pos - positions[i, :])
                
                X_GWO = (X1 + X2 + X3) / 3
                
                # استراتژی DLH ( Dimension-learning )
                random_wolf = positions[np.random.randint(self.n_wolves), :]
                X_DLH = positions[i, :] + np.random.uniform(-1, 1) * (alpha_pos - random_wolf)
                
                # ترکیب و جهش
                if stall_counter > 7:
                    # جهش بزرگ برای فرار از 92%
                    positions[i, :] = np.random.uniform(0, 1, self.n_features)
                else:
                    if np.random.rand() < 0.6:
                        positions[i, :] = X_GWO
                    else:
                        positions[i, :] = X_DLH

            positions = np.clip(positions, 0, 1)
            
            if (t+1) % 5 == 0 or t == 0:
                print(f"Iteration {t+1}/80 | Best Accuracy: {alpha_score:.4f} | Stall: {stall_counter}")

        final_indices = np.where(alpha_pos > 0.5)[0]
        return self.X.columns[final_indices].tolist(), alpha_score, history