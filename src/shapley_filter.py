import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier


def get_kSV_features(X, y, top_n=60, n_estimators=200, seed=42):
    print("Starting Phase 1: Fast Tree-SHAP Ranking...")

    model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=seed)
    model.fit(X, y)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_array = np.abs(np.array(shap_values))
            mean_shap = shap_array.mean(axis=(0, 1))
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                mean_shap = np.abs(shap_values).mean(axis=(0, 2))
            else:
                mean_shap = np.abs(shap_values).mean(axis=0)
        else:
            mean_shap = model.feature_importances_

    except Exception as e:
        print(f"⚠️ SHAP Warning: {e}. Switching to Model Feature Importances...")
        mean_shap = model.feature_importances_

    mean_shap = np.array(mean_shap).flatten()

    if len(mean_shap) != X.shape[1]:
        print("📏 Length mismatch! Recovering with model importances...")
        mean_shap = model.feature_importances_

    feature_importance = pd.DataFrame({
        'gene': X.columns.tolist(),
        'importance': mean_shap.tolist()
    }).sort_values(by='importance', ascending=False)

    top_n = min(top_n, X.shape[1])
    selected_genes = feature_importance.head(top_n)['gene'].tolist()
    print(f"Phase 1 complete. Top {len(selected_genes)} features selected successfully.")
    return selected_genes, feature_importance