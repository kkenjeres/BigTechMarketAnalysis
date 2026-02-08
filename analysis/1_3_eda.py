import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('Data/CleanedData.csv')

numeric_cols = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %', 'Spread']
corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title('Рисунок 2.1 – Матрица корреляций признаков', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('images/Figure_2_1_Correlation.png', dpi=150, bbox_inches='tight')
plt.close()
