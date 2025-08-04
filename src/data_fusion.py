import pandas as pd

def merge_data(df1, df4):
    df1 = df1.copy()
    df4 = df4.copy()

    df1['timestamp'] = pd.to_datetime(df1['timestamp'], unit='ms', errors='coerce')
    df4['timestamp'] = pd.to_datetime(df4['timestamp'], unit='ms', errors='coerce')

    df1 = df1.dropna(subset=['timestamp'])
    df4 = df4.dropna(subset=['timestamp'])
    df1 = df1.sort_values('timestamp').reset_index(drop=True)
    df4 = df4.sort_values('timestamp').reset_index(drop=True)

    df4['future_close_4h'] = df4['close'].shift(-1)

    df_merged = pd.merge_asof(
        df1,
        df4[['timestamp', 'future_close_4h']],
        on='timestamp',
        direction='backward',
        tolerance=pd.Timedelta('4h')
    )

    df_merged = df_merged.dropna(subset=['future_close_4h'])
    return df_merged