import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier

def get_kSV_features(X, y, top_n=60):
    print(f"Starting Phase 1: Fast Tree-SHAP Ranking...")
    
    # استفاده از تمام توان CPU شما
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X, y)
    
    try:
        explainer = shap.TreeExplainer(model)
        # محاسبه مقادیر شاپلی
        shap_values = explainer.shap_values(X)
        
        # --- مدیریت هوشمند انواع خروجی SHAP برای جلوگیری از خطای Length ---
        if isinstance(shap_values, list):
            # حالت چند کلاسه (لیستی از ماتریس‌ها)
            # تبدیل لیست به آرایه: (تعداد کلاس، تعداد نمونه، تعداد ژن)
            shap_array = np.abs(np.array(shap_values))
            # میانگین‌گیری روی محور کلاس (0) و محور نمونه (1) -> نتیجه: (تعداد ژن،)
            mean_shap = shap_array.mean(axis=(0, 1))
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                # خروجی جدید SHAP به صورت (نمونه، ژن، کلاس)
                mean_shap = np.abs(shap_values).mean(axis=(0, 2))
            else:
                # خروجی استاندارد 2 بعدی (نمونه، ژن)
                mean_shap = np.abs(shap_values).mean(axis=0)
        else:
            # اگر فرمت ناشناخته بود، از اهمیت داخلی مدل استفاده کن
            mean_shap = model.feature_importances_

    except Exception as e:
        print(f"⚠️ SHAP Warning: {e}. Switching to Model Feature Importances...")
        mean_shap = model.feature_importances_

    # اطمینان نهایی از اینکه طول آرایه با تعداد ستون‌ها یکی است
    mean_shap = np.array(mean_shap).flatten()
    
    if len(mean_shap) != X.shape[1]:
        print("📏 Length mismatch! Recovering with model importances...")
        mean_shap = model.feature_importances_

    # ساخت دیتابیس نهایی
    feature_importance = pd.DataFrame({
        'gene': X.columns.tolist(),
        'importance': mean_shap.tolist()
    }).sort_values(by='importance', ascending=False)

    selected_genes = feature_importance.head(top_n)['gene'].tolist()
    print(f"Phase 1 complete. Top {len(selected_genes)} features selected successfully.")
    return selected_genes, feature_importance