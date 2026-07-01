import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap

def get_kSV_features(X, y, top_n=100):
    """
    پیاده‌سازی فاز اول مقاله: رتبه‌بندی ویژگی‌ها با استفاده از منطق شاپلی
    """
    print(f"Starting Phase 1: kSV Feature Ranking...")
    
    # استفاده از RandomForest برای تخمین اولیه سریع اهمیت ژن‌ها
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # محاسبه Shapley Values
    # این بخش به مدل نگاه می‌کند و می‌گوید هر ژن چقدر در تشخیص سرطان سهم داشته است
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # مدیریت خروجی شاپ (بسته به نسخه کتابخانه)
    if isinstance(shap_values, list):
        mean_shap = np.abs(shap_values[1]).mean(axis=0)
    else:
        # اگر خروجی سه بعدی باشد (در نسخه‌های جدید)
        if len(shap_values.shape) == 3:
            mean_shap = np.abs(shap_values[:,:,1]).mean(axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)
        
    # ساخت دیتابیس از ژن‌ها و نمره اهمیت‌شان
    feature_importance = pd.DataFrame({
        'gene': X.columns,
        'importance': mean_shap
    }).sort_values(by='importance', ascending=False)
    
    # انتخاب تعداد مشخص شده از ژن‌های برتر (مثلا ۱۰۰ تا)
    selected_genes = feature_importance.head(top_n)['gene'].tolist()
    
    print(f"Phase 1 complete. Selected top {top_n} genes.")
    return selected_genes, feature_importance