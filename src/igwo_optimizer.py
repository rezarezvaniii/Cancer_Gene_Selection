import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class IGWO_Optimizer:
    def __init__(self, X, y, n_wolves=20, max_iter=30):
        self.X = X
        self.y = y
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.n_features = X.shape[1]
        
    def fitness_function(self, position):
        # تبدیل موقعیت پیوسته به باینری (ژن‌هایی که مقدارشان بالای 0.5 است انتخاب می‌شوند)
        selected_indices = np.where(position > 0.5)[0]
        
        if len(selected_indices) == 0:
            return 0
        
        # --- ایده نو: Ensemble Fitness (SVM + RF) ---
        clf_svm = SVC(kernel='poly', degree=3, C=1.0)
        clf_rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        
        X_subset = self.X.iloc[:, selected_indices]
        
        try:
            # محاسبه میانگین دقت با 3-fold cross-validation
            score_svm = cross_val_score(clf_svm, X_subset, self.y, cv=3).mean()
            score_rf = cross_val_score(clf_rf, X_subset, self.y, cv=3).mean()
            
            avg_acc = (score_svm + score_rf) / 2
            
            # جریمه برای تعداد ویژگی‌ها: هدف ما دقت بالا با کمترین ژن ممکن است
            # هرچه تعداد ژن‌ها کمتر باشد، جریمه کمتر و Fitness بیشتر می‌شود
            penalty = 0.01 * (len(selected_indices) / self.n_features)
            return avg_acc - penalty
        except:
            return 0

    def optimize(self):
        # مقداردهی اولیه جمعیت گرگ‌ها
        positions = np.random.uniform(0, 1, (self.n_wolves, self.n_features))
        
        alpha_pos = np.zeros(self.n_features)
        alpha_score = -np.inf
        
        beta_pos = np.zeros(self.n_features)
        beta_score = -np.inf
        
        delta_pos = np.zeros(self.n_features)
        delta_score = -np.inf

        history = [] # برای ذخیره روند همگرایی

        print(f"Starting Advanced IGWO (DLH + Ensemble Fitness)...")
        
        for t in range(self.max_iter):
            # 1. محاسبه Fitness برای تمام گرگ‌ها و تعیین رهبران
            for i in range(self.n_wolves):
                fitness = self.fitness_function(positions[i, :])
                
                if fitness > alpha_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = alpha_score, alpha_pos.copy()
                    alpha_score, alpha_pos = fitness, positions[i, :].copy()
                    
                elif fitness > beta_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = fitness, positions[i, :].copy()
                    
                elif fitness > delta_score:
                    delta_score, delta_pos = fitness, positions[i, :].copy()

            history.append(alpha_score)
            
            # پارامتر a که از 2 به 0 کاهش می‌یابد
            a = 2 - t * (2 / self.max_iter) 

            new_positions = np.zeros_like(positions)

            # 2. بروزرسانی موقعیت گرگ‌ها
            for i in range(self.n_wolves):
                for j in range(self.n_features):
                    # محاسبات حرکت به سمت Alpha, Beta, Delta (فاز GWO)
                    r1, r2 = np.random.random(), np.random.random()
                    A1, C1 = 2*a*r1 - a, 2*r2
                    D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                    X1 = alpha_pos[j] - A1 * D_alpha

                    r1, r2 = np.random.random(), np.random.random()
                    A2, C2 = 2*a*r1 - a, 2*r2
                    D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                    X2 = beta_pos[j] - A2 * D_beta

                    r1, r2 = np.random.random(), np.random.random()
                    A3, C3 = 2*a*r1 - a, 2*r2
                    D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                    X3 = delta_pos[j] - A3 * D_delta

                    X_GWO = (X1 + X2 + X3) / 3
                    
                    # فاز DLH (Dimension-Learning-Based Hunting) - استراتژی اصلی مقاله
                    # یادگیری ابعادی با استفاده از همسایگان تصادفی
                    idx1 = np.random.randint(self.n_wolves)
                    idx2 = np.random.randint(self.n_wolves)
                    X_DLH = positions[i, j] + np.random.random() * (positions[idx1, j] - positions[idx2, j])
                    
                    # ترکیب دو استراتژی برای جلوگیری از افتادن در تله بهینه محلی
                    if np.random.random() < 0.5:
                        new_positions[i, j] = X_GWO
                    else:
                        new_positions[i, j] = X_DLH

            positions = np.clip(new_positions, 0, 1)
            print(f"Iteration {t+1}/{self.max_iter} | Best Fitness: {alpha_score:.4f}")

        # استخراج نام ژن‌های نهایی
        final_indices = np.where(alpha_pos > 0.5)[0]
        selected_gene_names = self.X.columns[final_indices].tolist()
        
        return selected_gene_names, alpha_score, history