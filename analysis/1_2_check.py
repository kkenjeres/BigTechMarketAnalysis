import pandas as pd
import matplotlib.pyplot as plt


files = {
    'Amazon': 'data/AmazonPrice.csv',
    'Google': 'data/GooglePrice.csv',
    'Meta': 'data/MetaPrice.csv',
    'Microsoft': 'data/MicrosoftPrice.csv'
}

dfs = []
for name, path in files.items():
    d = pd.read_csv(path)
    d['Company'] = name
    dfs.append(d)

df = pd.concat(dfs, ignore_index=True)

total_rows = len(df)
total_nulls = df.isnull().sum().sum()

print(f"Всего строк: {total_rows}")
print(f"Пустых ячеек: {total_nulls}")
check_sample = df.isnull().sample(15, random_state=42)

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')
ax.set_title(f'Рисунок 3 – Результат проверки набора данных на чистоту', fontsize=14, fontweight='bold', y=1.05)

table = ax.table(cellText=check_sample.values, 
                 colLabels=check_sample.columns, 
                 loc='center', cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig('images/Figure_3_CheckTrueFalse.png', bbox_inches='tight', dpi=150)
plt.close()