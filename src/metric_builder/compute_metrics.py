import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.metric_builder.ecological_metrics import (
    compute_ei,
    compute_et,
    compute_network_greenness,
)
from src.metric_builder.network_metrics import (
    matrix_to_digraph,
    compute_centrality_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--base-path", default="data/parquet")
    parser.add_argument("--label-base-path", default="data/raw")
    parser.add_argument("--output-path", default="data/metrics")
    parser.add_argument("--min-weight", type=float, default=0)

    parser.add_argument(
        "--only-greenness",
        action="store_true",
        help="Only compute network-embedded green-ness using existing EI and ET files.",
    )

    return parser.parse_args()


def load_existing_ei_et(year, year_output):
    ei_path = year_output / f"ei_{year}.parquet"
    et_path = year_output / f"et_{year}.parquet"

    if not ei_path.exists():
        raise FileNotFoundError(f"Missing EI file: {ei_path}")

    if not et_path.exists():
        raise FileNotFoundError(f"Missing ET file: {et_path}")

    EI = pd.read_parquet(ei_path)["emissions_intensity"]
    ET = pd.read_parquet(et_path)

    return EI, ET


def compute_greenness_metric(year, year_output, EI, ET):
    print("Computing network-embedded green-ness...")

    greenness = compute_network_greenness(EI, ET)
    greenness.to_parquet(year_output / f"greenness_{year}.parquet")

    print(f"    -> Green-ness saved with columns: {list(greenness.columns)}")


def compute_year_metrics(
    year,
    base_path,
    label_base_path,
    output_path,
    min_weight,
    only_greenness=False,
):
    print(f"\n=== YEAR {year} ===")

    year_output = output_path / str(year)
    year_output.mkdir(parents=True, exist_ok=True)

    if only_greenness:
        print("[ONLY] Loading existing EI and ET...")
        EI, ET = load_existing_ei_et(year, year_output)
        compute_greenness_metric(year, year_output, EI, ET)
        print(f"=== DONE {year} ===\n")
        return

    print("[1] Computing EI...")
    EI = compute_ei(
        year,
        base_path=base_path,
        label_base_path=label_base_path,
    )
    EI.to_frame().to_parquet(year_output / f"ei_{year}.parquet")
    print("    -> EI saved")

    print("[2] Computing ET...")
    ET = compute_et(
        year,
        base_path=base_path,
        label_base_path=label_base_path,
    )
    ET.to_parquet(year_output / f"et_{year}.parquet")
    print(f"    -> ET saved (shape: {ET.shape})")

    print("[3] Computing network-embedded green-ness...")
    compute_greenness_metric(year, year_output, EI, ET)

    print("[4] Building graph...")
    G = matrix_to_digraph(
        ET,
        weight_col="embedded_emissions",
        min_weight=min_weight,
    )
    print(f"    -> Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

    print("[5] Computing centrality and PageRank...")
    centrality = compute_centrality_metrics(
        G,
        weight="embedded_emissions",
    )
    centrality.to_parquet(year_output / f"centrality_{year}.parquet")
    print(f"    -> Centrality saved with columns: {list(centrality.columns)}")

    print("[6] Computing efficiency metrics...")

    out_embodied = ET.sum(axis=1)
    in_embodied = ET.sum(axis=0)

    efficiency = centrality.copy()

    efficiency["out_embodied"] = out_embodied
    efficiency["in_embodied"] = in_embodied

    efficiency["out_efficiency"] = efficiency["out_embodied"] / efficiency["out_strength"]
    efficiency["in_efficiency"] = efficiency["in_embodied"] / efficiency["in_strength"]

    efficiency = efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)

    efficiency.to_parquet(year_output / f"efficiency_{year}.parquet")

    print("    -> Efficiency saved")

    print(f"=== DONE {year} ===\n")


def main():
    args = parse_args()

    base_path = Path(args.base_path)
    label_base_path = Path(args.label_base_path)
    output_path = Path(args.output_path)

    print("Starting metric computation")
    print(f"Years: {args.years}")
    print(f"Base path: {base_path}")
    print(f"Label path: {label_base_path}")
    print(f"Output path: {output_path}")
    print(f"Only green-ness: {args.only_greenness}")

    for year in args.years:
        compute_year_metrics(
            year=year,
            base_path=base_path,
            label_base_path=label_base_path,
            output_path=output_path,
            min_weight=args.min_weight,
            only_greenness=args.only_greenness,
        )

    print("All computations completed")


if __name__ == "__main__":
    main()