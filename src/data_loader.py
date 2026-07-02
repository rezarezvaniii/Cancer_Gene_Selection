import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from io import StringIO

def load_and_clean_data(filepath):
    print(f"Loading file: {filepath}")
    
    if filepath.endswith('.arff'):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        attributes = []
        data_start_idx = 0
        for i, line in enumerate(lines):
            clean_line = line.strip()
            if not clean_line: continue
            if clean_line.lower().startswith('@attribute'):
                parts = clean_line.split()
                attr_name = parts[1].strip("'").strip('"')
                attributes.append(attr_name)
            elif clean_line.lower().startswith('@data'):
                data_start_idx = i + 1
                break
        
        data_content = "".join(lines[data_start_idx:])
        df = pd.read_csv(StringIO(data_content), header=None, na_values='?')
        df.columns = attributes
    else:
        df = pd.read_csv(filepath, sep=None, engine='python')

    # حذف فضاهای خالی از نام ستون‌ها
    df.columns = df.columns.str.strip()

    # پیدا کردن ستون هدف: معمولاً آخرین ستونی است که عددی نیست
    target_col = None
    for col in reversed(df.columns):
        if df[col].dtype == object or df[col].dtype == bool:
            target_col = col
            break
    
    if target_col is None: target_col = df.columns[-1]
    
    print(f"Detected target column: '{target_col}'")

    # تبدیل مقادیر هدف به رشته و پاکسازی (مهم برای ARFF)
    y_raw = df[target_col].astype(str).str.strip().str.replace("b'", "").str.replace("'", "")
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # چاپ کلاس‌ها برای اطمینان
    unique_y = np.unique(y)
    print(f"Unique classes found: {unique_y} -> {le.classes_}")
    
    if len(unique_y) < 2:
        print("CRITICAL WARNING: Only one class detected! Manual search for target...")
        # اگر فقط یک کلاس پیدا شد، یعنی ستون هدف اشتباه انتخاب شده. ستون عددی آخر را تست می‌کنیم.
        y = le.fit_transform(df.iloc[:, -1])
    
    # استخراج ویژگی‌ها (فقط ستون‌های عددی)
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_final = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"Data loaded successfully! Shape: {X_final.shape}")
    return X_final, y, le.classes_