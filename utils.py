from pathlib import Path
import pandas as pd


def import_df(fname) -> pd.DataFrame:
    """Import dataframe from a given json file

    Args:
        fname (str): Json path
    """
    path = Path("data-generated") / fname
    assert path.exists(), "File does not exist"
    df = pd.read_json(path)
    return df


def export_df(df:pd.DataFrame, fname: str):
    dest_dir = Path("data-generated")
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / fname
    with open(dest, "w") as f:
        f.write(df.to_json())
