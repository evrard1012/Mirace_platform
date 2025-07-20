import pandas as pd

def description_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe()
