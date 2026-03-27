import pandas as pd


def load_foods(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
