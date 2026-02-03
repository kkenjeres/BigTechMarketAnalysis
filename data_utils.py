import pandas as pd
from sklearn.model_selection import train_test_split

RAW_FILES = {
    'Amazon': 'Data/AmazonPrice.csv',
    'Google': 'Data/GooglePrice.csv',
    'Meta': 'Data/MetaPrice.csv',
    'Microsoft': 'Data/MicrosoftPrice.csv',
}

CLEANED_PATH = 'Data/CleanedData.csv'
NUMERIC_COLS = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']


def load_raw():
    dfs = []
    for name, path in RAW_FILES.items():
        d = pd.read_csv(path)
        d['Company'] = name
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def _to_numeric(x):
    if isinstance(x, str):
        x = x.replace(',', '').replace('%', '')
        if 'M' in x:
            return float(x.replace('M', '')) * 1_000_000
        if 'B' in x:
            return float(x.replace('B', '')) * 1_000_000_000
        if 'K' in x:
            return float(x.replace('K', '')) * 1_000
        return float(x)
    return x


def clean_raw(df, save_path=None):
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_to_numeric)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Date', 'Company']).reset_index(drop=True)
    df['Company'] = df['Company'].astype('category').cat.codes
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Spread'] = df['High'] - df['Low']
    if save_path:
        df.to_csv(save_path, index=False)
    return df


def get_splits(test_size=0.2, shuffle=False, random_state=None):
    df = pd.read_csv(CLEANED_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Target'] = df.groupby('Company')['Price'].shift(-1) > df['Price']
    df['Target'] = df['Target'].fillna(False).astype(int)
    X = df.drop(columns=['Date', 'Company', 'Target'])
    y = df['Target']
    X = X.iloc[:-4]
    y = y.iloc[:-4]
    df = df.iloc[:-4]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, df
