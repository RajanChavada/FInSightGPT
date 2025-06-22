import pandas as pd

def safe_time_to_seconds(t):
    if pd.isna(t) or t in ['', '-', None]:
        return 0.0
    try:
        # convert to timedelta 
        if isinstance(t, str): 
            t = pd.to_timedelta(t)
        # If it's already a Timedelta → total_seconds
        if hasattr(t, 'total_seconds'):
            return t.total_seconds()
        # Else assume string "M:SS.sss" → parse
        if isinstance(t, str):
            m, s = t.split(":")
            return int(m) * 60 + float(s)
    except Exception as e:
        print(f"Failed to parse time: {t} ({e})")
        return 0.0



def build_driver_qualy_dict(df):
    """Create a nested dictionary of qualifying results by driver and race."""
    races = df['Race'].unique()
    drivers = df['Abbreviation'].unique()
    qualy = {}
    for race in races:
        for driver in drivers:
            row = df[(df['Abbreviation'] == driver) & (df['Race'] == race)]
            if row.empty:
                continue
            Q1 = safe_time_to_seconds(row['Q1'].values[0])
            Q2 = safe_time_to_seconds(row['Q2'].values[0])
            Q3 = safe_time_to_seconds(row['Q3'].values[0])
            if driver not in qualy:
                qualy[driver] = []
            qualy[driver].append({
                race: {'Q1': Q1, 'Q2': Q2, 'Q3': Q3}
            })
    return qualy

def qualy_dict_to_df(qualy):
    """Flatten the qualifying dictionary into a tabular DataFrame."""
    rows = []
    for driver, races in qualy.items():
        for race_data in races:
            for race, times in race_data.items():
                rows.append({
                    'Driver': driver,
                    'Race': race,
                    'Q1': times['Q1'],
                    'Q2': times['Q2'],
                    'Q3': times['Q3']
                })
    return pd.DataFrame(rows)

def normalize_column(series):
    """Min-max normalize a pandas Series."""
    min_val, max_val = series.min(), series.max()
    return (series - min_val) / (max_val - min_val) if max_val != min_val else pd.Series([0]*len(series), index=series.index)

def preprocess_data(df):
    """
    Final preprocessing step:
    - Create binary target column
    - Normalize Q1 times
    - One-hot encode Driver and Race
    """
    df = df.copy()

    # Convert Q1, Q2, Q3 to seconds first
    df['Q1'] = df['Q1'].apply(safe_time_to_seconds)
    df['Q2'] = df['Q2'].apply(safe_time_to_seconds)
    df['Q3'] = df['Q3'].apply(safe_time_to_seconds)

    df['made_q3'] = (df['Q3'] > 0).astype(int)
    df = df.drop(columns=['Q2', 'Q3'])
    df = df[df['Q1'] > 0]
    df['Q1_norm'] = normalize_column(df['Q1'])
    df = pd.get_dummies(df, columns=['Driver', 'Race'], drop_first=True)
    return df
