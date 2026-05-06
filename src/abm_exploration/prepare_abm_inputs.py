from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ABMInputConfig:
    metrics_dir: Path = Path("data/metrics")
    merged_panel_path: Path = Path("data/final/eora_atlas_merged.parquet")
    output_dir: Path = Path("data/abm")
    edge_quantile_threshold: float = 0.999
    top_edges_per_source: int | None = 25


class ABMInputBuilder:
    def __init__(self, config: ABMInputConfig) -> None:
        self.config = config

    def build_all(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        agents = self.build_agents_panel()
        transitions = self.build_transitions_panel(agents)
        edges = self.build_edges_panel()

        agents.to_parquet(self.config.output_dir / "agents_panel.parquet", index=False)
        transitions.to_parquet(self.config.output_dir / "transitions_panel.parquet", index=False)
        edges.to_parquet(self.config.output_dir / "edges_panel.parquet", index=False)

        logging.info("Saved agents panel: %s rows", len(agents))
        logging.info("Saved transitions panel: %s rows", len(transitions))
        logging.info("Saved edges panel: %s rows", len(edges))

    def build_agents_panel(self) -> pd.DataFrame:
        if self.config.merged_panel_path.exists():
            logging.info("Loading merged Eora-Atlas panel")
            df = pd.read_parquet(self.config.merged_panel_path)
        else:
            logging.info("Merged panel not found; building Eora-only agents panel")
            df = self._build_eora_only_agents_panel()

        df = self._standardize_agent_columns(df)
        df = self._add_regime_labels(df)

        return df.sort_values(["agent_id", "Year"]).reset_index(drop=True)

    def build_transitions_panel(self, agents: pd.DataFrame) -> pd.DataFrame:
        logging.info("Building transition panel")

        agents = agents.sort_values(["agent_id", "Year"]).copy()

        next_agents = agents.copy()
        next_agents["Year"] = next_agents["Year"] - 1

        keep_cols = [
            "agent_id",
            "Year",
            "emissions_intensity",
            "g_base",
            "g_out_network",
            "g_in_network",
            "pagerank",
            "out_strength",
            "in_strength",
            "regime",
        ]

        optional_cols = [
            "green_capability_share",
            "green_capability_export_share",
            "capability_mean_pci",
            "capability_export_weighted_pci",
        ]

        keep_cols += [col for col in optional_cols if col in agents.columns]
        keep_cols = [col for col in keep_cols if col in agents.columns]

        current = agents[keep_cols].copy()
        future = next_agents[keep_cols].copy()

        future = future.rename(
            columns={
                col: f"{col}_next"
                for col in future.columns
                if col not in ["agent_id", "Year"]
            }
        )

        transitions = current.merge(
            future,
            on=["agent_id", "Year"],
            how="inner",
            validate="one_to_one",
        )

        delta_cols = [
            "emissions_intensity",
            "g_base",
            "g_out_network",
            "g_in_network",
            "pagerank",
            "out_strength",
            "in_strength",
        ]

        for col in delta_cols:
            next_col = f"{col}_next"
            if col in transitions.columns and next_col in transitions.columns:
                transitions[f"delta_{col}"] = transitions[next_col] - transitions[col]

        if "regime_next" in transitions.columns:
            transitions["regime_transition"] = (
                transitions["regime"].astype(str)
                + " -> "
                + transitions["regime_next"].astype(str)
            )

        return transitions.replace([np.inf, -np.inf], np.nan)

    def build_edges_panel(self) -> pd.DataFrame:
        logging.info("Building sparse edges panel from ET matrices")

        frames: list[pd.DataFrame] = []

        for year_dir in self._year_dirs():
            year = int(year_dir.name)
            et_path = year_dir / f"et_{year}.parquet"

            if not et_path.exists():
                logging.warning("Skipping missing ET file: %s", et_path)
                continue

            logging.info("Loading ET for %s", year)
            et = pd.read_parquet(et_path)

            edges = self._matrix_to_sparse_edges(et, year)
            frames.append(edges)

        if not frames:
            raise ValueError("No ET edge panels could be built.")

        return pd.concat(frames, ignore_index=True)

    def _build_eora_only_agents_panel(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        for year_dir in self._year_dirs():
            year = int(year_dir.name)

            paths = {
                "ei": year_dir / f"ei_{year}.parquet",
                "greenness": year_dir / f"greenness_{year}.parquet",
                "centrality": year_dir / f"centrality_{year}.parquet",
                "efficiency": year_dir / f"efficiency_{year}.parquet",
            }

            missing = [name for name, path in paths.items() if not path.exists()]
            if missing:
                logging.warning("Skipping %s due to missing files: %s", year, missing)
                continue

            ei = pd.read_parquet(paths["ei"])
            greenness = pd.read_parquet(paths["greenness"])
            centrality = pd.read_parquet(paths["centrality"])
            efficiency = pd.read_parquet(paths["efficiency"])

            df = pd.concat([ei, greenness, centrality, efficiency], axis=1)
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.reset_index(names="country_sector")
            df["Year"] = year

            labels = self._split_country_sector(df["country_sector"])
            df = pd.concat([df, labels], axis=1)

            frames.append(df)

        if not frames:
            raise ValueError("No yearly metric panels could be built.")

        return pd.concat(frames, ignore_index=True)

    def _matrix_to_sparse_edges(self, et: pd.DataFrame, year: int) -> pd.DataFrame:
        values = et.to_numpy().ravel()
        positive_values = values[values > 0]

        if len(positive_values) == 0:
            return pd.DataFrame(
                columns=[
                    "Year",
                    "source_agent_id",
                    "target_agent_id",
                    "embedded_emissions",
                    "w_out",
                    "w_in",
                ]
            )

        threshold = float(
            np.quantile(positive_values, self.config.edge_quantile_threshold)
        )

        edges = et.stack().reset_index()
        edges.columns = ["source_agent_id", "target_agent_id", "embedded_emissions"]
        edges = edges[edges["embedded_emissions"] > threshold].copy()

        if self.config.top_edges_per_source is not None:
            edges = (
                edges.sort_values(
                    ["source_agent_id", "embedded_emissions"],
                    ascending=[True, False],
                )
                .groupby("source_agent_id", group_keys=False)
                .head(self.config.top_edges_per_source)
            )

        out_sums = edges.groupby("source_agent_id")["embedded_emissions"].transform("sum")
        in_sums = edges.groupby("target_agent_id")["embedded_emissions"].transform("sum")

        edges["w_out"] = edges["embedded_emissions"] / out_sums.replace(0, np.nan)
        edges["w_in"] = edges["embedded_emissions"] / in_sums.replace(0, np.nan)
        edges["Year"] = year

        return edges[
            [
                "Year",
                "source_agent_id",
                "target_agent_id",
                "embedded_emissions",
                "w_out",
                "w_in",
            ]
        ].replace([np.inf, -np.inf], np.nan).fillna(0)

    def _standardize_agent_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "country_sector" not in df.columns:
            df["country_sector"] = (
                df["Country"].astype(str)
                + " | "
                + df.get("Country_detail", "").astype(str)
                + " | "
                + df.get("Category", "").astype(str)
                + " | "
                + df["Sector"].astype(str)
            )

        df["agent_id"] = df["country_sector"].astype(str)

        numeric_cols = [
            "emissions_intensity",
            "g_base",
            "g_out_network",
            "g_in_network",
            "pagerank",
            "out_strength",
            "in_strength",
            "eigenvector_centrality",
            "reverse_eigenvector_centrality",
            "out_embodied",
            "in_embodied",
            "out_efficiency",
            "in_efficiency",
            "green_capability_share",
            "green_capability_export_share",
            "capability_mean_pci",
            "capability_export_weighted_pci",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Year"] = pd.to_numeric(df["Year"], errors="raise").astype(int)

        return df.replace([np.inf, -np.inf], np.nan)

    def _add_regime_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "g_base" not in df.columns or "pagerank" not in df.columns:
            df["regime"] = "unclassified"
            return df

        green_threshold = df["g_base"].median(skipna=True)
        centrality_threshold = df["pagerank"].median(skipna=True)

        df["green_status"] = np.where(
            df["g_base"] >= green_threshold,
            "green",
            "brown",
        )

        df["network_status"] = np.where(
            df["pagerank"] >= centrality_threshold,
            "core",
            "periphery",
        )

        df["regime"] = df["green_status"] + "_" + df["network_status"]

        return df

    def _split_country_sector(self, labels: pd.Series) -> pd.DataFrame:
        parts = labels.astype(str).str.split("|", expand=True)
        parts = parts.apply(lambda col: col.str.strip())

        if parts.shape[1] == 4:
            parts.columns = ["Country", "Country_detail", "Category", "Sector"]
            return parts

        logging.warning("Unexpected country-sector label format")
        return pd.DataFrame(
            {
                "Country": "",
                "Country_detail": "",
                "Category": "",
                "Sector": labels.astype(str),
            }
        )

    def _year_dirs(self) -> list[Path]:
        if not self.config.metrics_dir.exists():
            raise FileNotFoundError(f"Metrics directory not found: {self.config.metrics_dir}")

        return sorted(
            [
                path
                for path in self.config.metrics_dir.iterdir()
                if path.is_dir() and path.name.isdigit()
            ],
            key=lambda path: int(path.name),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ABM input panels from Remaster metric outputs."
    )

    parser.add_argument("--metrics-dir", default="data/metrics")
    parser.add_argument("--merged-panel-path", default="data/final/eora_atlas_merged.parquet")
    parser.add_argument("--output-dir", default="data/abm")
    parser.add_argument("--edge-quantile-threshold", type=float, default=0.999)
    parser.add_argument("--top-edges-per-source", type=int, default=25)

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    args = parse_args()

    top_edges_per_source = (
        None if args.top_edges_per_source <= 0 else args.top_edges_per_source
    )

    config = ABMInputConfig(
        metrics_dir=Path(args.metrics_dir),
        merged_panel_path=Path(args.merged_panel_path),
        output_dir=Path(args.output_dir),
        edge_quantile_threshold=args.edge_quantile_threshold,
        top_edges_per_source=top_edges_per_source,
    )

    builder = ABMInputBuilder(config)
    builder.build_all()


if __name__ == "__main__":
    main()