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

def clean_all(x):
    if isinstance(x, str):
        x = x.replace(',', '').replace('%', '')
        if 'M' in x: return float(x.replace('M', '')) * 1_000_000
        if 'B' in x: return float(x.replace('B', '')) * 1_000_000_000
        if 'K' in x: return float(x.replace('K', '')) * 1_000
        return float(x)
    return x

cols = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
for col in cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_all)
        df[col] = pd.to_numeric(df[col])

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Date', 'Company']).reset_index(drop=True)

df['Company'] = df['Company'].astype('category').cat.codes

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Spread'] = df['High'] - df['Low']

data_sample = df.groupby('Company').head(2).round(2)

fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('tight')
ax.axis('off')
ax.set_title(f'Рисунок 4 – Набор данных после очистки', fontsize=14, fontweight='bold', y=1.05)

table = ax.table(cellText=data_sample.values, 
                 colLabels=data_sample.columns, 
                 loc='center', cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.4)

plt.savefig('images/Figure_4_CleanedData.png', bbox_inches='tight', dpi=150)
plt.close()

df.to_csv('data/CleanedData.csv', index=False)