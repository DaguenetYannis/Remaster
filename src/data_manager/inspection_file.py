from pathlib import Path
import pandas as pd
import yaml
import csv

# -------------------------
# Load config
# -------------------------
CONFIG_PATH = "src/data_manager/config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

RAW_DIR = Path(config["paths"]["raw_dir"])

# -------------------------
# Settings
# -------------------------
YEAR = "1993"
VALID_MATRICES = {"FD", "Q", "QY", "T", "VA"}


def detect_delimiter(file_path: Path, n_bytes: int = 5000) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(n_bytes)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        return "\t"


def extract_matrix_name(file_path: Path) -> str | None:
    stem = file_path.stem  # e.g. Eora26_1993_bp_T
    parts = stem.split("_")
    if len(parts) >= 4 and parts[0] == "Eora26" and parts[1] == YEAR and parts[2] == "bp":
        return parts[3]
    return None


def inspect_file(file_path: Path) -> dict:
    delimiter = detect_delimiter(file_path)

    # Read once without header to inspect raw shape
    df_raw = pd.read_csv(
        file_path,
        sep=delimiter,
        header=None,
        encoding="utf-8",
        engine="python",
        on_bad_lines="warn"
    )

    # Read once with default header inference
    df_header = pd.read_csv(
        file_path,
        sep=delimiter,
        encoding="utf-8",
        engine="python",
        on_bad_lines="warn"
    )

    first_row = df_raw.iloc[0].tolist() if not df_raw.empty else []
    first_cols = list(df_header.columns)

    numeric_like_header = 0
    for col in first_cols:
        try:
            float(str(col))
            numeric_like_header += 1
        except Exception:
            pass

    header_guess = "likely_no_header" if numeric_like_header == len(first_cols) and len(first_cols) > 0 else "likely_has_header"

    return {
        "file": file_path.name,
        "delimiter": repr(delimiter),
        "raw_shape": df_raw.shape,
        "shape_with_header": df_header.shape,
        "header_guess": header_guess,
        "first_row_preview": first_row[:8],
        "column_preview": first_cols[:8],
    }


def main():
    year_dir = RAW_DIR / YEAR
    if not year_dir.exists():
        raise FileNotFoundError(f"Year folder not found: {year_dir}")

    txt_files = sorted(year_dir.rglob("*.txt"))

    selected_files = []
    for file_path in txt_files:
        matrix = extract_matrix_name(file_path)
        if matrix in VALID_MATRICES:
            selected_files.append((matrix, file_path))

    if not selected_files:
        print(f"No matching data files found for year {YEAR}.")
        return

    results = []
    print(f"Inspecting {len(selected_files)} data files for year {YEAR}...\n")

    for matrix, file_path in selected_files:
        info = inspect_file(file_path)
        info["matrix"] = matrix
        results.append(info)

        print(f"Matrix: {matrix}")
        print(f"File: {info['file']}")
        print(f"Delimiter: {info['delimiter']}")
        print(f"Raw shape: {info['raw_shape']}")
        print(f"Shape with header: {info['shape_with_header']}")
        print(f"Header guess: {info['header_guess']}")
        print(f"First row preview: {info['first_row_preview']}")
        print(f"Column preview: {info['column_preview']}")
        print("-" * 80)

    summary_df = pd.DataFrame(results)[[
        "matrix", "file", "delimiter", "raw_shape", "shape_with_header", "header_guess"
    ]]

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    # -------------------------
    # LABEL FILES (shapes only)
    # -------------------------
    print("\nLABEL FILES (shapes only)")
    print("=" * 80)

    for file_path in txt_files:
        name = file_path.name.lower()

        if not name.startswith("labels_"):
            continue

        try:
            df = pd.read_csv(
                file_path,
                sep="\t",
                header=None,
                encoding="utf-8",
                engine="python"
            )

            print(f"{file_path.name}: {df.shape}")

        except Exception as e:
            print(f"⚠️ Failed to read {file_path.name}: {e}")


if __name__ == "__main__":
    main()