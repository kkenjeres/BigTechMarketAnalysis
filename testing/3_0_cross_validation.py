import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from data_utils import get_splits
from design.models import all_models

X_train, X_test, y_train, y_test, _ = get_splits()
X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
y = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)

tscv = TimeSeriesSplit(n_splits=5)

results = {}
for name, model in all_models.items():
    scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
    mean_acc = scores.mean()
    std_acc = scores.std()
    results[name] = {'mean': mean_acc, 'std': std_acc, 'scores': scores}
    print(f"{name:22s}: {mean_acc:.4f} (+/- {std_acc:.4f})")

fig, ax = plt.subplots(figsize=(12, 7))
names = list(results.keys())
means = [results[n]['mean'] for n in names]
stds = [results[n]['std'] for n in names]

bars = ax.barh(names, means, xerr=stds, color='#00BFFF', edgecolor='black', capsize=5)
ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Случайное угадывание (50%)')
ax.set_xlabel('Средняя точность (Accuracy)', fontsize=12)
ax.set_title('Рисунок 10 – Кросс-валидация (TimeSeriesSplit, k=5)', fontsize=14, fontweight='bold')
ax.set_xlim(0.3, 0.7)
ax.legend(loc='lower right')
ax.invert_yaxis()

for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(mean + std + 0.01, i, f'{mean:.3f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('images/Figure_10_CrossValidation.png', dpi=150, bbox_inches='tight')
plt.close()
