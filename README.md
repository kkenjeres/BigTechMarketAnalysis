# BigTech Market Analysis

> Ensemble machine learning pipeline for predicting next-day stock price direction (up/down) using historical OHLC and volume data for Amazon, Google, Meta, and Microsoft.

---

## Overview

End-to-end ML project: load and preprocess market data, train multiple ensemble classifiers, evaluate with accuracy/ROC-AUC and k-fold cross-validation, tune hyperparameters, and export comparison plots. Target variable: binary (price tomorrow > price today).

---

## Features

- **Data pipeline**: Raw CSV load → missing-value check → numeric normalization (K/M/B, dates) → feature engineering (Year, Month, Day, Spread) → cleaned dataset
- **Models**: 7 classifiers — Random Forest, Gradient Boosting, AdaBoost, Extra Trees, Bagging, Voting (Hard), Voting (Soft)
- **Evaluation**: Accuracy, ROC-AUC, classification report; training time per model; 5-fold cross-validation
- **Optimization**: Hyperparameter search (e.g. `n_estimators` for Gradient Boosting)
- **Reproducibility**: Single shared module (`data_utils`) for load/clean/split; temporal train/test split (80/20, no shuffle)

---

## Tech stack

- **Python** 3
- **pandas** — data load and preprocessing
- **scikit-learn** — models, train/test split, cross-validation, metrics
- **matplotlib** — visualizations

---

## Setup

Clone the repo and install dependencies (from project root):

```bash
git clone <repo-url>
cd BigTechMarketAnalysis
pip3 install pandas matplotlib scikit-learn
```

Ensure `Data/` contains the input CSVs: `AmazonPrice.csv`, `GooglePrice.csv`, `MetaPrice.csv`, `MicrosoftPrice.csv`.

---

## Usage

Run from the **project root**. Order matters: generate the cleaned dataset first, then train and evaluate.

```bash
# 1. Analysis: load, check, metrics, clean → writes Data/CleanedData.csv
python3 analysis/1_1_load.py
python3 analysis/1_2_check.py
python3 analysis/1_2_metrics.py
python3 analysis/1_2_cleaned_data.py

# 2. Design: train/test split, train all models, save comparison plots
python3 design/2_2_split.py
python3 design/2_3_results.py

# 3. Testing: cross-validation, test metrics, hyperparameter optimization
python3 testing/3_0_cross_validation.py
python3 testing/3_1_metrics.py
python3 testing/3_2_optimization.py
```

Outputs: console metrics and training times; `images/` populated with Figure_1 through Figure_9; `Data/CleanedData.csv` after step 1.

---
