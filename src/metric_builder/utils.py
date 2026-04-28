from collections.abc import Iterable
from pathlib import Path
import pandas as pd
from typing import Dict, Tuple

# Matrix -> (row_label_file, column_label_file)
LABEL_RULES: Dict[str, Tuple[str, str]] = {
    "T": ("labels_T.txt", "labels_T.txt"),
    "Q": ("labels_Q.txt", "labels_T.txt"),
    "VA": ("labels_VA.txt", "labels_T.txt"),
    "FD": ("labels_T.txt", "labels_FD.txt"),
    "QY": ("labels_Q.txt", "labels_FD.txt"),
}

def read_parquet_file(parquet_file):
    """
    Reads a parquet file and returns a pandas DataFrame.

    Parameters:
    parquet_file (str): The path to the parquet file.

    Returns:
    pd.DataFrame: The DataFrame containing the data from the parquet file.
    """
    try:
        df = pd.read_parquet(parquet_file)
        return df
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return None


def read_label_file(label_file):
    """
    Reads a tab-separated label file and returns a pandas DataFrame.

    Parameters:
    label_file (str | Path): The path to the label file.

    Returns:
    pd.DataFrame: The DataFrame containing the label table.
    """
    try:
        return pd.read_csv(label_file, sep="\t", header=None, encoding="utf-8", engine="python")
    except Exception as e:
        print(f"Error reading label file: {e}")
        return None


def labels_to_strings(labels_df):
    """
    Turn each row of a label table into one readable string label.

    Parameters:
    labels_df (pd.DataFrame): The raw label table.

    Returns:
    pd.Index: A pandas index of joined label strings.
    """
    cleaned_labels = labels_df.fillna("").astype(str)
    joined_labels = cleaned_labels.apply(
        lambda row: " | ".join(value.strip() for value in row if value.strip() != ""),
        axis=1,
    )
    return pd.Index(joined_labels)


def apply_labels_to_matrix(df, row_labels_df, column_labels_df):
    """
    Apply row and column labels to a matrix dataframe.

    Parameters:
    df (pd.DataFrame): The matrix dataframe.
    row_labels_df (pd.DataFrame): The raw row label table.
    column_labels_df (pd.DataFrame): The raw column label table.

    Returns:
    pd.DataFrame: The labelled matrix dataframe.
    """
    labelled_df = df.copy()
    labelled_df.index = labels_to_strings(row_labels_df)
    labelled_df.columns = labels_to_strings(column_labels_df)
    return labelled_df


def load_matrices(years, matrices, base_path="."):
    if isinstance(years, int):
        years = [years]
    elif not isinstance(years, Iterable):
        raise TypeError("years must be an int or an iterable of ints")

    base_directory = Path(base_path)

    for year in years:
        for matrix in matrices:
            file_path = base_directory / str(year) / f"{matrix}.parquet"
            df = read_parquet_file(file_path)
            if df is not None:
                yield matrix.upper(), year, df


def load_labelled_matrices(years, matrices, base_path=".", label_base_path=None):
    """
    Load matrices with their row and column labels applied.

    Parameters:
    years (int | Iterable[int]): One year or a collection of years to load.
    matrices (Iterable[str]): Matrix names such as T, Q, QY, FD, VA.
    base_path (str | Path): Base folder containing yearly parquet folders.
    label_base_path (str | Path | None): Base folder containing yearly label files.
        If None, the same base path as the matrix files is used.

    Yields:
    tuple[str, int, pd.DataFrame]: Matrix name, year, and labelled dataframe.
    """
    if label_base_path is None:
        label_base_directory = Path(base_path)
    else:
        label_base_directory = Path(label_base_path)

    for matrix, year, df in load_matrices(years, matrices, base_path=base_path):
        if matrix not in LABEL_RULES:
            print(f"Warning: no label rule found for matrix {matrix}. Returning unlabelled matrix.")
            yield matrix, year, df
            continue

        row_label_file, column_label_file = LABEL_RULES[matrix]
        year_label_directory = label_base_directory / str(year)

        row_labels_df = read_label_file(year_label_directory / row_label_file)
        column_labels_df = read_label_file(year_label_directory / column_label_file)

        if row_labels_df is None or column_labels_df is None:
            print(f"Warning: could not load labels for {matrix} in {year}. Returning unlabelled matrix.")
            yield matrix, year, df
            continue

        labelled_df = apply_labels_to_matrix(df, row_labels_df, column_labels_df)
        yield matrix, year, labelled_df
