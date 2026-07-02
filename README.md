# Cancer Gene Selection using kSV-IGWO 🧬

This project implements a state-of-the-art feature selection method for cancer classification based on the 2025 research paper: **"Feature selection using game Shapley improved grey wolf optimizer for optimizing cancer classification"**.

## 🚀 Achievements

- **High Accuracy:** Reached **98.37%** on the Golub Leukemia dataset.
- **Dimensionality Reduction:** Reduced **7,129 genes** down to the **12 most impactful genes**.
- **Novelty:** Implemented an **Ensemble Fitness Function** (SVM + Random Forest) and **Multi-Objective Penalty** to ensure biological stability and minimal gene selection.

## 🛠️ Tech Stack

- **Language:** Python 3.10
- **Key Libraries:** `scikit-learn`, `SHAP` (Game Theory), `Seaborn`, `NumPy`.

## 📈 Methodology

1. **Phase 1 (kSV):** Ranked 7,000+ genes using Kernel Shapley Values to identify feature contributions.
2. **Phase 2 (IGWO):** Optimized the final gene subset using an Improved Grey Wolf Optimizer with Dimension-Learning-Based Hunting (DLH).
3. **Evaluation:** Validated using 10-fold Cross-Validation with a Polynomial SVM kernel.

## 📊 Results

The model successfully converged within 30 iterations, identifying a minimal set of genes that effectively distinguish between ALL and AML cancer types.
