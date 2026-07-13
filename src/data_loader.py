import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from io import StringIO

def load_and_clean_data(filepath):
    print(f"📂 Loading file: {filepath}")
    
    # ۱. خواندن فایل (پشتیبانی از ARFF و CSV)
    if filepath.endswith('.arff'):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        attributes = []
        data_start_idx = 0
        for i, line in enumerate(lines):
            clean_line = line.strip()
            if not clean_line: continue
            if clean_line.lower().startswith('@attribute'):
                parts = clean_line.split()
                # نام ویژگی را استخراج و کاراکترهای اضافه را پاک می‌کنیم
                attr_name = parts[1].strip("'").strip('"')
                attributes.append(attr_name)
            elif clean_line.lower().startswith('@data'):
                data_start_idx = i + 1
                break
        
        data_content = "".join(lines[data_start_idx:])
        df = pd.read_csv(StringIO(data_content), header=None, na_values='?')
        df.columns = attributes
    else:
        df = pd.read_csv(filepath)

    # ۲. پیدا کردن هوشمند ستون هدف (کلاس سرطان)
    target_col = None
    for col in df.columns:
        if 'class' in col.lower() or 'target' in col.lower():
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1] # اگر پیدا نشد، ستون آخر را فرض کن

    print(f"🎯 Target Column identified: '{target_col}'")

    # ۳. پاکسازی مقادیر ستون هدف (مهم برای SRBCT)
    # حذف کاراکترهای مزاحم مثل b' که در بعضی فایل‌های ARFF وجود دارد
    df[target_col] = df[target_col].astype(str).str.strip().str.replace("b'", "").str.replace("'", "")
    
    # ۴. حذف سطرهایی که برچسب (Label) نامعتبر دارند (رفع مشکل دقت صفر)
    invalid_values = ['nan', '?', 'none', 'null', '']
    # اینجا از .str.lower() استفاده شده تا خطای قبلی تکرار نشود
    df = df[~df[target_col].str.lower().isin(invalid_values)]

    # ۵. تبدیل برچسب‌های متنی به اعداد (Label Encoding)
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    
    # ۶. جدا کردن ویژگی‌ها (فقط ستون‌های عددی/ژن‌ها)
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])
    
    # ۷. مدیریت مقادیر خالی در ژن‌ها
    X = X.dropna(axis=1, how='all') # حذف ستون‌های کاملاً خالی
    X = X.fillna(X.mean())          # پر کردن جاهای خالی با میانگین همان ژن

    # ۸. نرمال‌سازی داده‌ها (استانداردسازی برای الگوریتم SVM و HHO)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_final = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"✅ Cleaned Classes: {np.unique(y)} -> {le.classes_}")
    print(f"✨ Final Data Shape: {X_final.shape}")
    
    return X_final, y, le.classes_