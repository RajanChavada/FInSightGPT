# utils.py
def normalize_column(series):
    min_val = series.min()
    max_val = series.max()
    return (series - min_val) / (max_val - min_val) if max_val != min_val else pd.Series([0] * len(series), index=series.index)
