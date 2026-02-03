import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from data_utils import get_splits

X_train, X_test, y_train, y_test, _ = get_splits()

X_tr_head = X_train.head(3).values
y_tr_head = y_train.head(3).values
X_te_head = X_test.head(2).values
y_te_head = y_test.head(2).values

np.set_printoptions(suppress=True, precision=1, linewidth=100, edgeitems=3)

text = ""
text += ">>> X_train, X_test, y_train, y_test = train_test_split(\n"
text += "...     X, y, test_size=0.2, shuffle=False)\n\n"
text += ">>> X_train (фрагмент)\n"
text += f"array({str(X_tr_head).replace('[ ', '[').replace('  ', ' ')})\n\n"
text += ">>> y_train (фрагмент)\n"
text += f"{str(y_tr_head)}\n\n"
text += ">>> X_test (фрагмент)\n"
text += f"array({str(X_te_head).replace('[ ', '[').replace('  ', ' ')})\n\n"
text += ">>> y_test (фрагмент)\n"
text += f"{str(y_te_head)}"

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')
rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, linewidth=2, edgecolor='black')
ax.add_patch(rect)
ax.text(0.05, 0.95, text, fontsize=10, fontfamily='monospace', verticalalignment='top', linespacing=1.5)
fig.text(0.5, 0.01, 'Рисунок 5 – Распечатки обучающей и тренировочной выборок', ha='center', fontsize=12, fontweight='bold')

plt.savefig('images/Figure_5_Train_Test_Split.png', bbox_inches='tight', dpi=150)
plt.close()
