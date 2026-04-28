from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml

# -------------------------
# Load config
# -------------------------
CONFIG_PATH = "src/data_manager/config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

RAW_DIR = Path(config["paths"]["raw_dir"])
PARQUET_DIR = Path(config["paths"].get("parquet_dir", "data/parquet"))
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Constants
# -------------------------
VALID_MATRICES = {"FD", "Q", "QY", "T", "VA"}

# matrix -> (row_label_file, col_label_file)
LABEL_RULES: Dict[str, Tuple[str, str]] = {
    "T": ("labels_T.txt", "labels_T.txt"),
    "Q": ("labels_Q.txt", "labels_T.txt"),
    "VA": ("labels_VA.txt", "labels_T.txt"),
    "FD": ("labels_T.txt", "labels_FD.txt"),
    "QY": ("labels_Q.txt", "labels_FD.txt"),
}


# -------------------------
# Path helpers
# -------------------------
def get_year_dir(year: int | str) -> Path:
    year_dir = RAW_DIR / str(year)
    if not year_dir.exists():
        raise FileNotFoundError(f"Year folder not found: {year_dir}")
    return year_dir


def get_output_year_dir(year: int | str) -> Path:
    output_dir = PARQUET_DIR / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_matrix_path(year: int | str, matrix: str) -> Path:
    matrix = matrix.upper()
    if matrix not in VALID_MATRICES:
        raise ValueError(f"Invalid matrix '{matrix}'. Expected one of {sorted(VALID_MATRICES)}")

    year_dir = get_year_dir(year)
    path = year_dir / f"Eora26_{year}_bp_{matrix}.txt"

    if not path.exists():
        raise FileNotFoundError(f"Matrix file not found: {path}")

    return path


def get_label_path(year: int | str, label_filename: str) -> Path:
    year_dir = get_year_dir(year)
    path = year_dir / label_filename

    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    return path


def get_output_matrix_path(year: int | str, matrix: str) -> Path:
    output_year_dir = get_output_year_dir(year)
    return output_year_dir / f"{matrix.upper()}.parquet"


# -------------------------
# Reading helpers
# -------------------------
def read_matrix(year: int | str, matrix: str) -> pd.DataFrame:
    path = get_matrix_path(year, matrix)
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        encoding="utf-8",
        engine="python",
    )


def read_label_file(year: int | str, label_filename: str) -> pd.DataFrame:
    path = get_label_path(year, label_filename)
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        encoding="utf-8",
        engine="python",
    )


def get_label_tables(year: int | str, matrix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = matrix.upper()
    if matrix not in LABEL_RULES:
        raise ValueError(f"No label rule defined for matrix '{matrix}'")

    row_label_file, col_label_file = LABEL_RULES[matrix]
    row_labels = read_label_file(year, row_label_file)
    col_labels = read_label_file(year, col_label_file)

    return row_labels, col_labels


# -------------------------
# Label formatting
# -------------------------
def labels_to_strings(labels_df: pd.DataFrame) -> pd.Index:
    """
    Turn each row of a label table into one readable string label.
    """
    cleaned = labels_df.fillna("").astype(str)
    joined = cleaned.apply(
        lambda row: " | ".join(x.strip() for x in row if x.strip() != ""),
        axis=1,
    )
    return pd.Index(joined)


# -------------------------
# Labelling and saving
# -------------------------
def apply_labels(year: int | str, matrix: str) -> pd.DataFrame:
    df = read_matrix(year, matrix)
    row_labels, col_labels = get_label_tables(year, matrix)

    df.index = labels_to_strings(row_labels)
    df.columns = labels_to_strings(col_labels)

    return df


def save_matrix_to_parquet(year: int | str, matrix: str, overwrite: bool = False) -> Path:
    matrix = matrix.upper()
    output_path = get_output_matrix_path(year, matrix)

    if output_path.exists() and not overwrite:
        print(f"⏭️ Skipping {year} {matrix} (already exists)")
        return output_path

    print(f"📦 Processing {year} {matrix}...")
    df = apply_labels(year, matrix)
    df.to_parquet(output_path)

    print(f"✅ Saved: {output_path}")
    return output_path


def process_year(year: int | str, overwrite: bool = False) -> None:
    print(f"\n===== Year {year} =====")
    for matrix in ["FD", "Q", "QY", "T", "VA"]:
        try:
            save_matrix_to_parquet(year, matrix, overwrite=overwrite)
        except Exception as e:
            print(f"⚠️ Failed for {year} {matrix}: {e}")


def process_all_years(start_year: int = 1990, end_year: int = 2017, overwrite: bool = False) -> None:
    for year in range(start_year, end_year + 1):
        process_year(year, overwrite=overwrite)


# -------------------------
# Main
# -------------------------
def main():
    process_all_years(overwrite=False)


if __name__ == "__main__":
    main()