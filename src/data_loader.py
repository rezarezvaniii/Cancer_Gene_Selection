import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from io import StringIO


def load_and_clean_data(filepath):
    print(f"📂 Loading file: {filepath}")

    if filepath.endswith('.arff'):
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        attributes = []
        data_start_idx = 0
        for i, line in enumerate(lines):
            clean_line = line.strip()
            if not clean_line:
                continue
            if clean_line.lower().startswith('@attribute'):
                parts = clean_line.split()
                attr_name = parts[1].strip("'").strip('"')
                attributes.append(attr_name)
            elif clean_line.lower().startswith('@data'):
                data_start_idx = i + 1
                break

        data_content = "".join(lines[data_start_idx:])
        df = pd.read_csv(StringIO(data_content), header=None, na_values='?')
        # اگه تعداد ستون‌های داده با تعداد @attribute ها یکی نبود (بعضی فایل‌های خراب)
        if df.shape[1] == len(attributes):
            df.columns = attributes
        else:
            df.columns = [f"col_{i}" for i in range(df.shape[1])]
            print(f"⚠️ Warning: attribute count mismatch ({len(attributes)} vs {df.shape[1]} cols). "
                  f"Using generic column names.")
    elif filepath.lower().endswith(('.xlsx', '.xls')):
        # اول یه نگاه به ردیف اول می‌ندازیم تا ببینیم واقعاً هدره یا داده‌ی خام
        probe = pd.read_excel(filepath, header=None, nrows=1)
        first_row = probe.iloc[0].tolist()

        def _looks_numeric(v):
            try:
                float(v)
                return True
            except (TypeError, ValueError):
                return False

        header_missing = all(_looks_numeric(v) for v in first_row)
        if header_missing:
            print("⚠️ Warning: first row of this Excel file looks like numeric data, not "
                  "column names — reading with no header so no sample is lost. "
                  "Generic column names (col_0, col_1, ...) will be used; the LAST column "
                  "is assumed to hold the class label (change this manually if that's wrong).")
            df = pd.read_excel(filepath, header=None)
            df.columns = [f"col_{i}" for i in range(df.shape[1])]
        else:
            df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    # اسم ستون‌ها رو به رشته تبدیل می‌کنیم (فایل‌های اکسل بدون هدر مناسب ممکنه
    # اسم ستون رو به‌صورت عدد صحیح (int) برگردونن که .lower() روش کار نمی‌کنه)
    df.columns = [str(c) for c in df.columns]

    target_col = None
    for col in df.columns:
        if 'class' in col.lower() or 'target' in col.lower():
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]

    # --- بررسی عقلانی: یک ستون کلاس واقعی باید تعداد مقادیر یکتای کمی داشته باشه
    # (مثلاً چند دسته‌ی بیماری)، نه تقریباً یک مقدار متفاوت برای هر نمونه. اگه ستون
    # انتخاب‌شده این‌طوری نبود، یعنی احتمالاً یک ستون ژنی پیوسته رو اشتباهی به‌عنوان
    # کلاس برداشتیم؛ پس دنبال ستون بهتری می‌گردیم و اگه پیدا نشد، خطای واضح می‌دیم
    # به‌جای ادامه‌ی خاموش با نتایج بی‌معنی (دقت ۰٪).
    max_reasonable_classes = max(10, int(len(df) * 0.2))
    if df[target_col].nunique() > max_reasonable_classes:
        print(f"⚠️ Warning: column '{target_col}' has {df[target_col].nunique()} unique values "
              f"for only {len(df)} rows — too many to be a real class label. "
              f"Searching other columns for a better candidate...")
        candidates = [
            c for c in df.columns
            if c != target_col and df[c].nunique() <= max_reasonable_classes
        ]
        if candidates:
            # اولویت با ستون آخر (چون در دیتاست‌های این حوزه معمولاً کلاس آخرین ستونه)
            target_col = candidates[-1]
            print(f"   ↪️  Using '{target_col}' instead ({df[target_col].nunique()} unique values).")
        else:
            raise ValueError(
                f"Could not automatically find a valid class-label column in '{filepath}'. "
                f"Every column has too many unique values to be a category. This usually means "
                f"the file is missing its class column, or the data is transposed (samples as "
                f"columns instead of rows). Please check the raw file manually."
            )

    print(f"🎯 Target Column identified: '{target_col}'")

    df[target_col] = (
        df[target_col].astype(str).str.strip()
        .str.replace("b'", "", regex=False)
        .str.replace("'", "", regex=False)
    )

    invalid_values = ['nan', '?', 'none', 'null', '']
    df = df[~df[target_col].str.lower().isin(invalid_values)]

    le = LabelEncoder()
    y = le.fit_transform(df[target_col])

    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])

    X = X.dropna(axis=1, how='all')
    X = X.fillna(X.mean())

    # حذف ژن‌هایی که واریانس صفر دارن (هیچ اطلاعاتی نمی‌دن و باعث خطای SVM می‌شن)
    zero_var_cols = X.columns[X.std(axis=0) == 0]
    if len(zero_var_cols) > 0:
        X = X.drop(columns=zero_var_cols)
        print(f"🧹 Removed {len(zero_var_cols)} zero-variance genes")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_final = pd.DataFrame(X_scaled, columns=X.columns)

    print(f"✅ Cleaned Classes: {np.unique(y)} -> {le.classes_}")
    print(f"✨ Final Data Shape: {X_final.shape}")

    # هشدار برای کلاس‌های خیلی کوچیک (باعث مشکل در CV می‌شه)
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < 5:
        print(f"⚠️ Warning: smallest class has only {counts.min()} samples — "
              f"CV folds will be reduced automatically.")

    return X_final, y, le.classes_