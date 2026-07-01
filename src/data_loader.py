import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_clean_data(filepath):
    print(f"Loading file: {filepath}")
    
    # استفاده از sep=None اجازه می‌دهد پانداز خودش تشخیص دهد جداکننده Tab است یا Comma یا Space
    try:
        df = pd.read_csv(filepath, sep=None, engine='python')
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None

    # حذف فاصله‌های خالی احتمالی از اول و آخر نام ستون‌ها
    df.columns = df.columns.str.strip()
    
    print("Columns found in file:", df.columns.tolist()[:10]) # برای عیب‌یابی

    if 'cancer' not in df.columns:
        raise KeyError("Column 'cancer' not found! Please check your CSV file column names.")

    # جدا کردن Target
    y = df['cancer'].values
    
    # پیدا کردن ستون‌هایی که داده‌های ژنی هستند (معمولاً از ستون ۶ به بعد)
    # یا ستون‌هایی که با 'AFFX' یا اسامی مشابه شروع می‌شوند
    # اینجا فرض می‌کنیم ۵ ستون اول اطلاعات جانبی هستند
    X = df.drop(columns=['Samples', 'BM.PB', 'Gender', 'Source', 'tissue.mf', 'cancer'], errors='ignore')
    
    # تبدیل برچسب‌ها به عدد
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # نرمال‌سازی
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_final = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"Data loaded successfully! Shape: {X_final.shape}")
    return X_final, y, le.classes_