import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/CleanedData.csv')
df['Date'] = pd.to_datetime(df['Date'])


df['Target'] = df.groupby('Company')['Price'].shift(-1) > df['Price']
df['Target'] = df['Target'].fillna(False).astype(int)


X = df.drop(columns=['Date', 'Company', 'Target'])
y = df['Target']

X = X.iloc[:-4] 
y = y.iloc[:-4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

X_tr_head = X_train.head(3).values
y_tr_head = y_train.head(3).values
X_te_head = X_test.head(2).values
y_te_head = y_test.head(2).values

np.set_printoptions(suppress=True, precision=1, linewidth=100, edgeitems=3)

text = ""
text += ">>> X_train, X_test, y_train, y_test = train_test_split(\n"
text += "...     X, y, test_size=0.2, shuffle=False)\n\n"

text += ">>> X_train (фрагмент)\n"
arr_str = str(X_tr_head).replace('[ ', '[').replace('  ', ' ')
text += f"array({arr_str})\n\n"

text += ">>> y_train (фрагмент)\n"
text += f"{str(y_tr_head)}\n\n"

text += ">>> X_test (фрагмент)\n"
arr_str_test = str(X_te_head).replace('[ ', '[').replace('  ', ' ')
text += f"array({arr_str_test})\n\n"

text += ">>> y_test (фрагмент)\n"
text += f"{str(y_te_head)}"

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, linewidth=2, edgecolor='black')
ax.add_patch(rect)

ax.text(0.05, 0.95, text, 
        fontsize=10, 
        fontfamily='monospace', 
        verticalalignment='top', 
        linespacing=1.5)

fig.text(0.5, 0.01, 'Рисунок 5 – Распечатки обучающей и тренировочной выборок', 
         ha='center', fontsize=12, fontweight='bold')

plt.savefig('images/Figure_5_Train_Test_Split.png', bbox_inches='tight', dpi=150)
plt.close()