import pandas as pd


def load_foods(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["category"] = df["category"].fillna("(unknown)").astype(str)
    return df.reset_index(drop=True)
