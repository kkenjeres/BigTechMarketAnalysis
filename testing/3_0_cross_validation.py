"""
Запускать из корня проекта: python3 testing/3_0_cross_validation.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from data_utils import get_splits
from design.models import all_models

X_train, X_test, y_train, y_test, _ = get_splits()
X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
y = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)

cv = KFold(n_splits=5, shuffle=False, random_state=None)
print("K-fold кросс-валидация (k=5), метрика: accuracy")
print("=" * 60)

for name, model in all_models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    mean_acc = scores.mean()
    std_acc = scores.std()
    print(f"{name}: {mean_acc:.4f} (+/- {std_acc:.4f})  [fold scores: {scores.round(4)}]")

print("=" * 60)
print("Итоговая эффективность модели должна оцениваться на отложенной тестовой выборке (см. 3_1_metrics, 2_3_results).")
