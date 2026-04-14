import zipfile
from pathlib import Path
import shutil
import tempfile
import duckdb

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output_parquet")
LOG_DIR = Path("logs")

SUPPORTED_EXTENSIONS = {".csv", ".txt", ".tsv"}

OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / "zip_to_parquet.log"


def log(message: str) -> None:
    print(message)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def convert_file_to_parquet(input_file: Path, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ext = input_file.suffix.lower()

    if ext == ".csv":
        query = f"""
            COPY (
                SELECT * FROM read_csv_auto('{input_file.as_posix()}', header=true)
            )
            TO '{output_file.as_posix()}' (FORMAT PARQUET)
        """
    elif ext == ".tsv":
        query = f"""
            COPY (
                SELECT * FROM read_csv_auto(
                    '{input_file.as_posix()}',
                    delim='\t',
                    header=true
                )
            )
            TO '{output_file.as_posix()}' (FORMAT PARQUET)
        """
    elif ext == ".txt":
        query = f"""
            COPY (
                SELECT * FROM read_csv_auto('{input_file.as_posix()}', header=true)
            )
            TO '{output_file.as_posix()}' (FORMAT PARQUET)
        """
    else:
        raise ValueError(f"Unsupported extension: {ext}")

    con = duckdb.connect()
    try:
        con.execute(query)
    finally:
        con.close()


def process_zip(zip_path: Path) -> None:
    zip_stem = zip_path.stem
    log(f"\nProcessing ZIP: {zip_path.name}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]

            for member in members:
                inner_path = Path(member.filename)

                if inner_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    log(f"  Skipped unsupported file: {member.filename}")
                    continue

                try:
                    extracted_path = tmpdir_path / inner_path
                    extracted_path.parent.mkdir(parents=True, exist_ok=True)

                    with zf.open(member) as source, extracted_path.open("wb") as target:
                        shutil.copyfileobj(source, target)

                    parquet_relative = inner_path.with_suffix(".parquet")
                    parquet_output = OUTPUT_DIR / zip_stem / parquet_relative

                    convert_file_to_parquet(extracted_path, parquet_output)
                    log(f"  Converted: {member.filename} -> {parquet_output}")

                except Exception as e:
                    log(f"  Failed: {member.filename} | Error: {e}")


def main() -> None:
    if log_file.exists():
        log_file.unlink()

    zip_files = sorted(DATA_DIR.glob("*.zip"))

    if not zip_files:
        log("No ZIP files found in data/")
        return

    for zip_path in zip_files:
        process_zip(zip_path)

    log("\nDone.")


if __name__ == "__main__":
    main()