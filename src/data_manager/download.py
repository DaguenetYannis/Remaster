import requests
import yaml
from pathlib import Path
from tqdm import tqdm
from urllib.parse import quote

# -------------------------
# Load config
# -------------------------
CONFIG_PATH = "src/data_manager/config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

RAW_DIR = Path(config["paths"]["raw_dir"])
RAW_DIR.mkdir(parents=True, exist_ok=True)

EMAIL = config["credentials"]["email"]
PASSWORD = config["credentials"]["password"]

ENCODED_EMAIL = quote(EMAIL)
ENCODED_PASSWORD = quote(PASSWORD)

# -------------------------
# Constants
# -------------------------
BASE_URL = "https://worldmrio.com/ComputationsM/Phase199/Loop082/simplified/Eora26_{}_bp.zip"
YEARS = range(1990, 2018)
LOG_FILE = RAW_DIR / "download_log.txt"


# -------------------------
# Helpers
# -------------------------
def already_downloaded(file_path):
    return file_path.exists() and file_path.stat().st_size > 0


def log_download(year):
    with open(LOG_FILE, "a") as f:
        f.write(f"{year}\n")


def build_url(year):
    return (
        BASE_URL.format(year)
        + f"?email={ENCODED_EMAIL}&pass={ENCODED_PASSWORD}"
    )


def download_file(url, output_path):
    response = requests.get(url, stream=True)

    if response.status_code != 200:
        print(f"❌ Failed ({response.status_code}): {url}")
        return False

    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as file, tqdm(
        desc=output_path.name,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

    return True


# -------------------------
# Main
# -------------------------
def main():
    print("🚀 Starting download of Eora26 bp files...\n")

    for year in YEARS:
        filename = f"Eora26_{year}_bp.zip"
        output_path = RAW_DIR / filename
        url = build_url(year)

        if already_downloaded(output_path):
            print(f"⏭️ Skipping {year} (already downloaded)")
            continue

        print(f"⬇️ Downloading {year}...")

        success = download_file(url, output_path)

        if success:
            print(f"✅ Completed {year}\n")
            log_download(year)
        else:
            print(f"⚠️ Skipped {year} due to error\n")

    print("🎉 All done!")


if __name__ == "__main__":
    main()