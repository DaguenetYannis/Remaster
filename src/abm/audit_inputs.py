from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_NODE_COLUMNS = [
    "Year",
    "country_sector",
    "Country",
    "Sector",
    "X",
    "D",
    "M",
    "EI",
    "g_local",
    "g_in",
    "g_out",
    "NG",
    "inventory_base",
    "capacity_base",
    "capability_readiness",
]

OPTIONAL_NODE_COLUMNS = [
    "out_strength",
    "in_strength",
    "pagerank",
    "eigenvector_centrality",
    "reverse_eigenvector_centrality",
]


def audit_metrics_panel(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    print("\n=== METRICS PANEL ===")
    print("Path:", path)
    print("Shape:", df.shape)
    print("Years:", sorted(df["Year"].dropna().unique().tolist()))

    missing_required = [col for col in REQUIRED_NODE_COLUMNS if col not in df.columns]
    print("Missing required columns:", missing_required)

    records = []

    for year, group in df.groupby("Year"):
        record = {
            "Year": year,
            "rows": len(group),
            "duplicate_country_sector": group["country_sector"].duplicated().sum(),
        }

        for col in REQUIRED_NODE_COLUMNS + OPTIONAL_NODE_COLUMNS:
            if col not in group.columns:
                record[f"{col}_missing_col"] = True
                continue

            if col in ["Year", "country_sector", "Country", "Sector"]:
                continue

            values = pd.to_numeric(group[col], errors="coerce")
            record[f"{col}_nan"] = int(values.isna().sum())
            record[f"{col}_inf"] = int(np.isinf(values).sum())
            record[f"{col}_zero"] = int((values == 0).sum())

        records.append(record)

    report = pd.DataFrame(records)
    return report


def audit_et_alignment(metrics_panel_path: Path, metrics_dir: Path) -> pd.DataFrame:
    panel = pd.read_parquet(metrics_panel_path)
    panel["Year"] = pd.to_numeric(panel["Year"], errors="coerce").astype("Int64")

    records = []

    print("\n=== ET ALIGNMENT ===")

    for year in sorted(panel["Year"].dropna().unique().tolist()):
        year = int(year)
        et_path = metrics_dir / str(year) / f"et_{year}.parquet"

        year_panel = panel[panel["Year"] == year].copy()
        expected = pd.Index(year_panel["country_sector"].astype(str))

        record = {
            "Year": year,
            "panel_rows": len(year_panel),
            "et_exists": et_path.exists(),
        }

        if not et_path.exists():
            records.append(record)
            continue

        et = pd.read_parquet(et_path)
        et.index = et.index.astype(str)
        et.columns = et.columns.astype(str)

        record["et_rows"] = et.shape[0]
        record["et_cols"] = et.shape[1]
        record["index_intersection"] = len(set(et.index) & set(expected))
        record["column_intersection"] = len(set(et.columns) & set(expected))
        record["index_duplicates"] = int(et.index.duplicated().sum())
        record["column_duplicates"] = int(et.columns.duplicated().sum())

        try:
            aligned = et.loc[expected, expected]
            record["alignable"] = True
            record["aligned_nan"] = int(aligned.isna().sum().sum())
            record["positive_values"] = int((aligned.to_numpy() > 0).sum())
        except Exception as exc:
            record["alignable"] = False
            record["alignment_error"] = str(exc)

        records.append(record)

    return pd.DataFrame(records)


def main() -> None:
    metrics_panel_path = Path("data/abm/metrics/abm_metrics_panel.parquet")
    metrics_dir = Path("data/metrics")
    output_dir = Path("data/abm/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_report = audit_metrics_panel(metrics_panel_path)
    et_report = audit_et_alignment(metrics_panel_path, metrics_dir)

    metrics_report.to_csv(output_dir / "abm_metrics_panel_audit.csv", index=False)
    et_report.to_csv(output_dir / "et_alignment_audit.csv", index=False)

    print("\n=== METRICS AUDIT SUMMARY ===")
    print(metrics_report[["Year", "rows", "duplicate_country_sector"]].head())

    print("\n=== ET AUDIT SUMMARY ===")
    print(et_report.head())

    print("\nSaved:")
    print(output_dir / "abm_metrics_panel_audit.csv")
    print(output_dir / "et_alignment_audit.csv")


if __name__ == "__main__":
    main()