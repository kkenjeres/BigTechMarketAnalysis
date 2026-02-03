import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from data_utils import get_splits

X_train, X_test, y_train, y_test, _ = get_splits()

param_range = range(10, 305, 10)
scores = []

for n in param_range:
    model = GradientBoostingClassifier(n_estimators=n, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    scores.append(acc)

plt.figure(figsize=(10, 6))
plt.scatter(param_range, scores, color='#444444', label='Результаты тестов', alpha=0.8)
best_idx = np.argmax(scores)
best_x = param_range[best_idx]
best_y = scores[best_idx]
plt.scatter([best_x], [best_y], color='#FFD700', s=100, label=f'Лучший результат ({best_y:.2f})', edgecolors='black')
plt.title('Рисунок 9 – Оптимизация количества деревьев (Gradient Boosting)', fontsize=14, fontweight='bold')
plt.xlabel('Значение гиперпараметра (Количество деревьев, n_estimators)', fontsize=11)
plt.ylabel('Оценка точности (Accuracy)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')
plt.savefig('images/Figure_9_Optimization.png', bbox_inches='tight', dpi=150)
plt.close()
