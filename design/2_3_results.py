import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from data_utils import get_splits
from design.models import all_models

X_train, X_test, y_train, y_test, _ = get_splits()

results = {}
trained_models = {}

for name, model in all_models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    trained_models[name] = model
    print(f"{name}: {acc:.2%}")

plt.figure(figsize=(12, 7))
slice_len = 60
X_plot = X_test.iloc[:slice_len]

for name, model in trained_models.items():
    if name == 'Voting (Hard)':
        continue
    probs = model.predict_proba(X_plot)[:, 1]
    plt.plot(range(len(X_plot)), probs, label=name, alpha=0.6, linewidth=1.5)

plt.title('Рисунок 6 – Визуализация итоговых моделей', fontsize=14, fontweight='bold')
plt.ylabel('Вероятность роста (Confidence)', fontsize=12)
plt.xlabel('Номер торгового дня (из тестового периода)', fontsize=12)
plt.ylim(0, 1.05)
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')
plt.savefig('images/Figure_6_ModelDynamics.png', bbox_inches='tight', dpi=150)
plt.close()

plt.figure(figsize=(10, 6))
sorted_res = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
plt.barh(list(sorted_res.keys()), list(sorted_res.values()), color='#00BFFF', height=0.6)
plt.axvline(x=0.5, color='red', linestyle='--')
plt.xlim(0, 0.7)
plt.title('Рисунок 7 – Итоговая точность моделей (Accuracy)', fontsize=12, fontweight='bold')
plt.gca().invert_yaxis()
for i, v in enumerate(sorted_res.values()):
    plt.text(v + 0.005, i, f'{v:.4f}', va='center', fontweight='bold')
plt.savefig('images/Figure_7_GlobalComparison.png', bbox_inches='tight', dpi=150)
plt.close()
