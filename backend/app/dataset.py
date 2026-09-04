import pandas as pd
from datasets import load_dataset

DATASET_NAME = "lang-uk/recruitment-dataset-job-descriptions-english"


def load_raw_jobs() -> pd.DataFrame:
    """Download the dataset via `datasets`."""
    df = load_dataset(DATASET_NAME, split="train").to_pandas()
    assert isinstance(df, pd.DataFrame)
    return df


def filter_jobs(df: pd.DataFrame, keywords: list[str], min_desc_chars: int = 200) -> pd.DataFrame:
    """Filter by Primary Keyword in keywords, drop empty/too-short descriptions. Pure."""
    keyword_mask = df["Primary Keyword"].isin(keywords)
    length_mask = df["Long Description"].str.len() >= min_desc_chars
    return df[keyword_mask & length_mask]


def deduplicate_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """Dedup by (Company Name, Position), case-insensitive/whitespace-normalized comparison,
    keeping the first-seen row with its original casing. Pure."""
    dedup_key = (
        df["Company Name"].str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
        + "\x00"
        + df["Position"].str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    )
    return df[~dedup_key.duplicated()]
