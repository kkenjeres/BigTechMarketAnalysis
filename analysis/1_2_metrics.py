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

def force_numeric(x):
    if isinstance(x, str):
        x = x.replace(',', '').replace('%', '')
        if 'M' in x: return float(x.replace('M', '')) * 1_000_000
        if 'B' in x: return float(x.replace('B', '')) * 1_000_000_000
        if 'K' in x: return float(x.replace('K', '')) * 1_000
    return x

cols = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
for col in cols:
    if col in df.columns:
        df[col] = df[col].apply(force_numeric)
        df[col] = pd.to_numeric(df[col], errors='coerce')

df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Spread'] = df['High'] - df['Low']

numeric_df = df.drop(columns=['Date', 'Company'])

metrics = numeric_df.describe().round(2)

fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('tight')
ax.axis('off')
ax.set_title('Рисунок 2 – Стандартные метрики исходного набора данных', fontsize=14, fontweight='bold', y=1.05)

table = ax.table(cellText=metrics.values, 
                 colLabels=metrics.columns, 
                 rowLabels=metrics.index, 
                 loc='center', cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.4)

plt.savefig('images/Figure_2_Metrics.png', bbox_inches='tight', dpi=150)
plt.close()