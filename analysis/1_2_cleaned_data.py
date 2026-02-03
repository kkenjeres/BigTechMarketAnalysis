import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from data_utils import load_raw, clean_raw, CLEANED_PATH

df = load_raw()
df = clean_raw(df, save_path=CLEANED_PATH)

data_sample = df.groupby('Company').head(2).round(2)

fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('tight')
ax.axis('off')
ax.set_title('Рисунок 4 – Набор данных после очистки', fontsize=14, fontweight='bold', y=1.05)

table = ax.table(cellText=data_sample.values, colLabels=data_sample.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.4)

plt.savefig('images/Figure_4_CleanedData.png', bbox_inches='tight', dpi=150)
plt.close()
