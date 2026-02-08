# BigTech Market Analysis

An ensemble machine learning pipeline for analyzing and predicting stock price movements (Growth or Fall) for major technology companies: Amazon, Google, Meta, and Microsoft.

---

## 📋 Project Overview

This research project focuses on a binary classification task: predicting the direction of stock price movement for the next trading day based on historical OHLCV data (Open, High, Low, Close, Volume).

### Key Features:

- **Comprehensive Pipeline**: Complete data lifecycle from raw CSV ingestion to cleaned dataset generation.
- **Ensemble Learning**: Implementation of 9 model configurations, including Random Forest, Gradient Boosting, AdaBoost, Extra Trees, Bagging, and Voting Classifiers.
- **In-depth Analysis**: Correlation matrices, TimeSeriesSplit cross-validation, training time measurement, and hyperparameter optimization.
- **Automated Visualization**: Generates 11 unique plots covering EDA, model training, and performance evaluation.

---

## 🛠 Tech Stack

- **Python 3.13+**
- **Pandas** — Data manipulation and analysis.
- **Scikit-learn** — Machine learning models and validation.
- **Matplotlib / Seaborn** — Result visualization.

---

## 🚀 Usage Guide

For accurate execution, run the scripts from the project root in the following sequence:

### 1. Data Preparation

Generates the processed `Data/CleanedData.csv` file.

```bash
python3 analysis/1_1_load.py
python3 analysis/1_2_check.py
python3 analysis/1_2_metrics.py
python3 analysis/1_2_cleaned_data.py
python3 analysis/1_3_eda.py
```

### 2. Model Training

Creates comparison plots and measures execution efficiency.

```bash
python3 design/2_2_split.py
python3 design/2_3_results.py
```

### 3. Testing & Validation

```bash
python3 testing/3_0_cross_validation.py
python3 testing/3_1_metrics.py
python3 testing/3_2_optimization.py
```

---

## 📊 Visual Outputs

All plots are exported to the `images/` directory:

- `Figure_2_1_Correlation.png` — Feature correlation matrix.
- `Figure_7_GlobalComparison.png` — Model accuracy comparison.
- `Figure_10_CrossValidation.png` — Cross-validation stability results.
- `Figure_9_Optimization.png` — Hyperparameter tuning results.

---

