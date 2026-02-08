import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from data_utils import get_splits
from design.models import all_models

X_train, X_test, y_train, y_test, _ = get_splits()

results = {}
train_times = {}
trained_models = {}


for name, model in all_models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    
    train_times[name] = elapsed
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    trained_models[name] = model
    print(f"{name:22s}: Accuracy {acc:.2%}, Время обучения: {elapsed:.4f} сек")

print("="*70)

print("\n" + "="*70)
print("СВОДКА РЕЗУЛЬТАТОВ")
print("="*70)
sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)
print(f"{'Модель':<24} {'Accuracy':>10} {'Время (сек)':>12}")
print("-"*48)
for name, acc in sorted_res:
    print(f"{name:<24} {acc:>10.4f} {train_times[name]:>12.4f}")
print("="*70)

plt.figure(figsize=(12, 7))
slice_len = 60
X_plot = X_test.iloc[:slice_len]

for name, model in trained_models.items():
    if name in ['Voting (Hard)', 'Logistic Regression']:
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
sorted_res_dict = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
plt.barh(list(sorted_res_dict.keys()), list(sorted_res_dict.values()), color='#00BFFF', height=0.6)
plt.axvline(x=0.5, color='red', linestyle='--')
plt.xlim(0, 0.7)
plt.title('Рисунок 7 – Итоговая точность моделей (Accuracy)', fontsize=12, fontweight='bold')
plt.gca().invert_yaxis()
for i, v in enumerate(sorted_res_dict.values()):
    plt.text(v + 0.005, i, f'{v:.4f}', va='center', fontweight='bold')
plt.savefig('images/Figure_7_GlobalComparison.png', bbox_inches='tight', dpi=150)
plt.close()

