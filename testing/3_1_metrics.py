import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

if not os.path.exists('images'):
    os.makedirs('images')

df = pd.read_csv('data/CleanedData.csv')
df['Date'] = pd.to_datetime(df['Date'])

df['Target'] = df.groupby('Company')['Price'].shift(-1) > df['Price']
df['Target'] = df['Target'].fillna(False).astype(int)
df['Company_Code'] = df['Company']

X = df.drop(columns=['Date', 'Company', 'Target'])
y = df['Target']

X = X.iloc[:-4]
y = y.iloc[:-4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
et = ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42)

vote = VotingClassifier(estimators=[('rf', rf), ('gb', gb), ('et', et)], voting='soft')
vote.fit(X_train, y_train)

y_pred = vote.predict(X_test)
y_proba = vote.predict_proba(X_test)[:, 1] 

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred)


text_output = ""
text_output += ">>> from sklearn.metrics import accuracy_score, roc_auc_score\n"
text_output += ">>> # Расчет эффективности модели на тестовой выборке\n"
text_output += f">>> accuracy_score(y_test, y_pred)\n"
text_output += f"{acc:.5f}\n\n"
text_output += f">>> roc_auc_score(y_test, y_pred_proba)\n"
text_output += f"{roc:.5f}\n\n"
text_output += ">>> print(classification_report(y_test, y_pred))\n"
text_output += report

plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.axis('off')

plt.text(0.02, 0.98, text_output, 
         fontsize=11, 
         fontfamily='monospace', 
         verticalalignment='top',
         linespacing=1.3)

rect = plt.Rectangle((0.01, 0.01), 0.98, 0.98, fill=False, linewidth=2, edgecolor='black')
ax.add_patch(rect)

plt.title('Рисунок 8 – Результаты тестирования (метрики эффективности)', fontsize=12, fontweight='bold', y=1.02)
plt.savefig('images/Figure_8_Testing_Results.png', bbox_inches='tight', dpi=150)
plt.close()