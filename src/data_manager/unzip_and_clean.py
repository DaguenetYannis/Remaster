import shutil
import zipfile
from pathlib import Path
import yaml


# -------------------------
# Load config
# -------------------------
CONFIG_PATH = "src/data_manager/config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

RAW_DIR = Path(config["paths"]["raw_dir"])
SAVED_ZIPS_DIR = RAW_DIR / "saved zips"


# -------------------------
# Helpers
# -------------------------
def ensure_directories():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_ZIPS_DIR.mkdir(parents=True, exist_ok=True)


def extract_year_from_name(zip_path: Path):
    """
    Expect filenames like: Eora26_1990_bp.zip
    Returns '1990' if matched, else None.
    """
    name = zip_path.stem  # Eora26_1990_bp
    parts = name.split("_")
    if len(parts) == 3 and parts[0] == "Eora26" and parts[2] == "bp":
        year = parts[1]
        if year.isdigit():
            return year
    return None


def unzip_to_year_folder(zip_path: Path, year: str):
    """
    Extract zip contents into RAW_DIR/year
    """
    target_dir = RAW_DIR / year
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    return target_dir


def delete_import_subfolders(base_dir: Path):
    """
    Recursively delete every folder named 'import' under base_dir.
    """
    deleted = 0
    for path in base_dir.rglob("*"):
        if path.is_dir() and path.name.lower() == "import":
            shutil.rmtree(path)
            deleted += 1
            print(f"🗑️ Deleted: {path}")
    return deleted


def move_zip_to_archive(zip_path: Path):
    destination = SAVED_ZIPS_DIR / zip_path.name
    shutil.move(str(zip_path), str(destination))
    print(f"📦 Moved zip to: {destination}")


# -------------------------
# Main
# -------------------------
def main():
    ensure_directories()

    zip_files = sorted(RAW_DIR.glob("Eora26_*_bp.zip"))

    if not zip_files:
        print("No matching zip files found in data/raw.")
        return

    total_deleted_imports = 0

    for zip_path in zip_files:
        year = extract_year_from_name(zip_path)

        if not year:
            print(f"⚠️ Skipping unexpected filename: {zip_path.name}")
            continue

        print(f"\n📂 Processing {zip_path.name} for year {year}")

        year_dir = RAW_DIR / year
        if year_dir.exists():
            print(f"ℹ️ Folder already exists: {year_dir}")
        else:
            unzip_to_year_folder(zip_path, year)
            print(f"✅ Extracted to: {year_dir}")

        deleted_count = delete_import_subfolders(year_dir)
        total_deleted_imports += deleted_count

        move_zip_to_archive(zip_path)

    print("\n🎉 Done.")
    print(f"Total import folders deleted: {total_deleted_imports}")


if __name__ == "__main__":
    main()