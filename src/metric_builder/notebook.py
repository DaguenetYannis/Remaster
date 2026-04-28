import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path

    return Path, pd


@app.cell
def _(Path):
    metrics_path = Path("data\metrics")
    return (metrics_path,)


@app.cell
def _(metrics_path, pd):
    ei_data = {}
    et_data = {}
    centrality_data = {}

    for year_folder in metrics_path.iterdir():
        if not year_folder.is_dir():
            continue

        year = year_folder.name

        ei_path = year_folder / f"ei_{year}.parquet"
        et_path = year_folder / f"et_{year}.parquet"
        centrality_path = year_folder / f"centrality_{year}.parquet"

        if ei_path.exists():
            ei_data[year] = pd.read_parquet(ei_path)

        if et_path.exists():
            et_data[year] = pd.read_parquet(et_path)

        if centrality_path.exists():
            centrality_data[year] = pd.read_parquet(centrality_path)
    return centrality_data, ei_data, et_data


@app.cell
def _(ei_data):
    ei_data["1990"]
    return


@app.cell
def _(et_data):
    et_data["1990"]
    return


@app.cell
def _(centrality_data):
    centrality_data["1990"]
    return


if __name__ == "__main__":
    app.run()
