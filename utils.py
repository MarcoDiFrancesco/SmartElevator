from pathlib import Path
import pandas as pd


def import_df(fname) -> pd.DataFrame:
    """Import dataframe from a given json file

    Args:
        fname (str): Json path
    """
    fname = Path(fname)
    assert fname.exists()
    df = pd.read_json(fname)
    # df = df.set_index("index")
    # df["time"] = pd.to_datetime(df["time"], unit='ms')
    return df
