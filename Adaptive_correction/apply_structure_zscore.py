import pandas as pd

def load_zscore_params(csv_path):
    df = pd.read_csv(csv_path).set_index('feature')
    return {row: (df.loc[row, 'mean'], df.loc[row, 'std']) for row in df.index}

def apply_zscore(df, params):
    df = df.copy()
    for col, (mu, std) in params.items():
        if col in df.columns:
            df[col] = (df[col] - mu) / std
    return df
