import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from data_utils import load_raw, clean_raw

df = load_raw()
df = clean_raw(df)
numeric_df = df.drop(columns=['Date', 'Company'])
metrics = numeric_df.describe().round(2)

fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('tight')
ax.axis('off')
ax.set_title('Рисунок 2 – Стандартные метрики исходного набора данных', fontsize=14, fontweight='bold', y=1.05)

table = ax.table(cellText=metrics.values, colLabels=metrics.columns, rowLabels=metrics.index, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.4)

plt.savefig('images/Figure_2_Metrics.png', bbox_inches='tight', dpi=150)
plt.close()
