import pandas as pd
import numpy as np
import shap
from sklearn.svm import SVC

def get_kSV_features(X, y, top_n=60):
    """
    پیاده‌سازی دقیق فاز اول مقاله (kSV):
    استفاده از Kernel SHAP برای محاسبه سهم هر ژن و انتخاب 60 ژن برتر.
    """
    print(f"Starting Phase 1: kSV (Kernel Shapley) Feature Ranking...")

    # 1. تعریف مدل مطابق مقاله (استفاده از SVM خطی برای تخمین اهمیت ژن‌ها در فاز اول)
    # ما از kernel='linear' استفاده می‌کنیم چون برای داده‌های پزشکی پر ابعاد بهترین عملکرد رو داره
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)

    # 2. تعریف KernelExplainer (مطابق متد kSV مقاله)
    # نکته: چون تعداد ژن‌ها زیاده، محاسبات شاپلی خیلی سنگین میشه.
    # برای حل این مشکل، از میانگین داده‌ها (Background) استفاده می‌کنیم.
    background_data = shap.sample(X, 10) # 10 نمونه برای تخمین پس‌زمینه کافیست
    
    # تعریف تابع پیش‌بینی (احتمال کلاس مثبت)
    def predict_function(x):
        return model.predict_proba(x)[:, 1]

    explainer = shap.KernelExplainer(predict_function, background_data)

    # 3. محاسبه مقادیر شاپلی برای کل داده‌ها (یا بخشی از آن‌ها برای سرعت بیشتر)
    # در اینجا برای دقت بالا، روی کل X محاسبه می‌کنیم
    print("Calculating SHAP values (this may take a few minutes)...")
    shap_values = explainer.shap_values(X)

    # 4. محاسبه میانگین ارزش مطلق شاپلی برای هر ژن
    # این عدد نشان‌دهنده "میزان تأثیر" هر ژن در نتیجه نهایی است
# در فایل shapley_filter.py بخش مدیریت خروجی شاپ:
    if isinstance(shap_values, list):
        # تبدیل لیست به آرایه و میانگین‌گیری روی تمام کلاس‌ها (مخصوص SRBCT)
        shap_array = np.array(shap_values)
        mean_shap = np.abs(shap_array).mean(axis=(0, 1))
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)

    # 5. ساخت جدول رتبه‌بندی
    feature_importance = pd.DataFrame({
        'gene': X.columns,
        'importance': mean_shap
    }).sort_values(by='importance', ascending=False)

    # 6. انتخاب دقیقاً 60 ژن برتر (طبق Table 3 مقاله)
    selected_genes = feature_importance.head(top_n)['gene'].tolist()

    print(f"Phase 1 complete. Top {top_n} genes selected via kSV.")
    return selected_genes, feature_importance