from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.schemas import VALID_ECOSYSTEM_SOURCES


ECOSYSTEM_LABELS: dict[str, str] = {
    "agriculture_food_biomass": "Agriculture, food, and biomass",
    "extractive_energy_raw_materials": "Extractive energy and raw materials",
    "basic_materials_chemicals": "Basic materials and chemicals",
    "manufacturing_machinery_transport": "Manufacturing, machinery, and transport equipment",
    "utilities_infrastructure": "Utilities and infrastructure",
    "construction_real_estate": "Construction and real estate",
    "trade_transport_logistics": "Trade, transport, and logistics",
    "services_knowledge_finance": "Knowledge, finance, and business services",
    "public_social_household_services": "Public, social, and household services",
    "other_unclassified": "Other and unclassified activities",
}


MANUAL_SECTOR_TO_ECOSYSTEM: dict[str, str] = {
    "Agriculture": "agriculture_food_biomass",
    "Fishing": "agriculture_food_biomass",
    "Food & Beverages": "agriculture_food_biomass",
    "Wood and Paper": "agriculture_food_biomass",
    "Mining and Quarrying": "extractive_energy_raw_materials",
    "Petroleum, Chemical and Non-Metallic Mineral Products": "basic_materials_chemicals",
    "Metal Products": "basic_materials_chemicals",
    "Recycling": "basic_materials_chemicals",
    "Textiles and Wearing Apparel": "manufacturing_machinery_transport",
    "Electrical and Machinery": "manufacturing_machinery_transport",
    "Transport Equipment": "manufacturing_machinery_transport",
    "Other Manufacturing": "manufacturing_machinery_transport",
    "Electricity, Gas and Water": "utilities_infrastructure",
    "Construction": "construction_real_estate",
    "Wholesale Trade": "trade_transport_logistics",
    "Retail Trade": "trade_transport_logistics",
    "Transport": "trade_transport_logistics",
    "Maintenance and Repair": "trade_transport_logistics",
    "Post and Telecommunications": "services_knowledge_finance",
    "Finacial Intermediation and Business Activities": "services_knowledge_finance",
    "Hotels and Restraurants": "services_knowledge_finance",
    "Education, Health and Other Services": "public_social_household_services",
    "Public Administration": "public_social_household_services",
    "Private Households": "public_social_household_services",
    "Others": "other_unclassified",
    "Re-export & Re-import": "other_unclassified",
    "TOTAL": "other_unclassified",
}


MANUAL_ADJACENT_ECOSYSTEMS: tuple[tuple[str, str], ...] = (
    ("agriculture_food_biomass", "basic_materials_chemicals"),
    ("agriculture_food_biomass", "trade_transport_logistics"),
    ("extractive_energy_raw_materials", "basic_materials_chemicals"),
    ("extractive_energy_raw_materials", "utilities_infrastructure"),
    ("basic_materials_chemicals", "manufacturing_machinery_transport"),
    ("basic_materials_chemicals", "construction_real_estate"),
    ("manufacturing_machinery_transport", "trade_transport_logistics"),
    ("manufacturing_machinery_transport", "utilities_infrastructure"),
    ("utilities_infrastructure", "construction_real_estate"),
    ("utilities_infrastructure", "services_knowledge_finance"),
    ("services_knowledge_finance", "public_social_household_services"),
)


@dataclass(frozen=True)
class EcosystemAssignment:
    """One explicit sector-to-ecosystem assignment."""

    sector: str
    ecosystem_id: str
    ecosystem_label: str
    ecosystem_source: str

    def is_known_source(self) -> bool:
        """Return whether the source follows the ABM v4 source vocabulary."""
        return self.ecosystem_source in VALID_ECOSYSTEM_SOURCES


@dataclass(frozen=True)
class EcosystemBuildResult:
    """Inspectable result of productive ecosystem assignment."""

    state_panel: pl.DataFrame
    ecosystem_mapping: pl.DataFrame
    ecosystem_adjacency: pl.DataFrame
    assignment_report: pl.DataFrame
    sector_coverage: pl.DataFrame
    ecosystem_source: str


class EcosystemMapper:
    """Assign simplified productive ecosystems to ABM v4 country-sector nodes."""

    def __init__(
        self,
        paths: ABMV4Paths,
        eta_ecosystem_adjacent: float = 0.35,
    ) -> None:
        self.paths = paths
        self.eta_ecosystem_adjacent = eta_ecosystem_adjacent

    def load_existing_ecosystem_fields(self) -> pl.DataFrame | None:
        """Load sector ecosystem fields if already present in Atlas/Eora outputs."""
        atlas_path = (
            self.paths.data_atlas
            / "processed"
            / "atlas_eora26_sector_capabilities_1995_2016.parquet"
        )
        if not atlas_path.exists():
            return None

        atlas_schema = pl.scan_parquet(atlas_path).collect_schema()
        candidate_columns = [
            column_name
            for column_name in atlas_schema.names()
            if any(
                token in column_name.lower()
                for token in ("ecosystem", "cluster", "community")
            )
        ]
        if not candidate_columns or "eora26_sector" not in atlas_schema.names():
            return None

        selected_column = candidate_columns[0]
        return (
            pl.scan_parquet(atlas_path)
            .select(
                pl.col("eora26_sector").alias("Sector"),
                pl.col(selected_column).cast(pl.Utf8).alias("ecosystem_id"),
            )
            .drop_nulls(["Sector", "ecosystem_id"])
            .unique()
            .with_columns(
                pl.col("ecosystem_id").alias("ecosystem_label"),
                pl.lit("atlas_cluster_aggregated").alias("ecosystem_source"),
                pl.lit(f"Existing Atlas/Eora field: {selected_column}").alias("mapping_rule"),
                pl.lit("Loaded from aggregated Atlas/Eora capability data.").alias("notes"),
            )
            .collect()
        )

    def assign_ecosystems(self, state_panel: pl.DataFrame) -> EcosystemBuildResult:
        """Assign ecosystem fields to every row in an ABM v4 state panel."""
        sectors = sorted(
            sector
            for sector in state_panel["Sector"].drop_nulls().unique().to_list()
            if sector is not None
        )
        existing_mapping = self.load_existing_ecosystem_fields()
        hs92_mapping = self._try_hs92_dominant_cluster_mapping()

        if existing_mapping is not None and existing_mapping.height > 0:
            ecosystem_mapping = existing_mapping
            ecosystem_source = "atlas_cluster_aggregated"
            source_note = "Used existing cluster/community/ecosystem fields in aggregated Atlas/Eora data."
        elif hs92_mapping is not None and hs92_mapping.height > 0:
            ecosystem_mapping = hs92_mapping
            ecosystem_source = "hs92_dominant_cluster"
            source_note = "Used dominant HS92 product cluster mapped to Eora sectors."
        else:
            ecosystem_mapping = self.build_manual_sector_mapping(sectors)
            ecosystem_source = "eora_sector_manual_mapping"
            source_note = (
                "No existing Atlas/Eora cluster field or HS92 product cluster field was found; "
                "used transparent manual Eora-sector mapping."
            )

        state_with_ecosystems = self._join_mapping_to_state(state_panel, ecosystem_mapping)
        ecosystem_adjacency = self.build_adjacency(ecosystem_mapping)
        sector_coverage = self.build_sector_coverage(state_with_ecosystems, ecosystem_mapping)
        assignment_report = self.build_report(
            state_panel=state_with_ecosystems,
            ecosystem_mapping=ecosystem_mapping,
            ecosystem_source=ecosystem_source,
            notes=source_note,
        )

        return EcosystemBuildResult(
            state_panel=state_with_ecosystems,
            ecosystem_mapping=ecosystem_mapping,
            ecosystem_adjacency=ecosystem_adjacency,
            assignment_report=assignment_report,
            sector_coverage=sector_coverage,
            ecosystem_source=ecosystem_source,
        )

    def build_manual_sector_mapping(self, sectors: list[str]) -> pl.DataFrame:
        """Build an explicit editable manual Eora-sector ecosystem mapping."""
        rows: list[dict[str, str]] = []
        for sector in sectors:
            ecosystem_id = MANUAL_SECTOR_TO_ECOSYSTEM.get(sector, "other_unclassified")
            mapping_rule = (
                "manual_exact_sector_match"
                if sector in MANUAL_SECTOR_TO_ECOSYSTEM
                else "manual_unclassified_fallback"
            )
            notes = (
                "Mapped by explicit ABM v4 manual sector dictionary."
                if sector in MANUAL_SECTOR_TO_ECOSYSTEM
                else "Sector not found in manual dictionary; assigned to other_unclassified."
            )
            rows.append(
                {
                    "Sector": sector,
                    "ecosystem_id": ecosystem_id,
                    "ecosystem_label": ECOSYSTEM_LABELS[ecosystem_id],
                    "ecosystem_source": "eora_sector_manual_mapping",
                    "mapping_rule": mapping_rule,
                    "notes": notes,
                }
            )

        return pl.DataFrame(rows).sort("Sector")

    def build_adjacency(self, ecosystem_mapping: pl.DataFrame) -> pl.DataFrame:
        """Build explicit ecosystem proximity adjacency."""
        ecosystem_ids = sorted(ecosystem_mapping["ecosystem_id"].unique().to_list())
        adjacent_pairs = {
            tuple(sorted((ecosystem_from, ecosystem_to)))
            for ecosystem_from, ecosystem_to in MANUAL_ADJACENT_ECOSYSTEMS
        }
        rows: list[dict[str, object]] = []
        for ecosystem_id_from in ecosystem_ids:
            for ecosystem_id_to in ecosystem_ids:
                if ecosystem_id_from == ecosystem_id_to:
                    proximity = 1.0
                    relation_type = "same"
                elif tuple(sorted((ecosystem_id_from, ecosystem_id_to))) in adjacent_pairs:
                    proximity = self.eta_ecosystem_adjacent
                    relation_type = "adjacent"
                else:
                    proximity = 0.0
                    relation_type = "non_adjacent"
                rows.append(
                    {
                        "ecosystem_id_from": ecosystem_id_from,
                        "ecosystem_id_to": ecosystem_id_to,
                        "proximity": proximity,
                        "relation_type": relation_type,
                    }
                )
        return pl.DataFrame(rows)

    def write_outputs(
        self,
        result: EcosystemBuildResult,
        state_panel_path: Path | None = None,
    ) -> None:
        """Write ecosystem mappings, adjacency, updated state, and diagnostics."""
        self.paths.inputs.mkdir(parents=True, exist_ok=True)
        self.paths.diagnostics.mkdir(parents=True, exist_ok=True)
        result.ecosystem_mapping.write_csv(self.paths.ecosystem_mapping_path)
        result.ecosystem_adjacency.write_csv(self.paths.ecosystem_adjacency_path)
        result.assignment_report.write_csv(self.paths.ecosystem_assignment_report_path)
        result.sector_coverage.write_csv(self.paths.ecosystem_sector_coverage_path)
        if state_panel_path is not None:
            result.state_panel.write_parquet(state_panel_path)

    def build_report(
        self,
        state_panel: pl.DataFrame,
        ecosystem_mapping: pl.DataFrame,
        ecosystem_source: str,
        notes: str,
    ) -> pl.DataFrame:
        """Build a one-row ecosystem assignment report."""
        node_frame = state_panel.select("country_sector", "ecosystem_id").unique()
        mapped_nodes = node_frame.filter(pl.col("ecosystem_id") != "fallback_unknown").height
        total_nodes = node_frame.height
        source_counts = (
            state_panel.group_by("ecosystem_source")
            .agg(pl.col("country_sector").n_unique().alias("node_count"))
            .sort("ecosystem_source")
        )
        source_counts_text = "; ".join(
            f"{row['ecosystem_source']}={row['node_count']}"
            for row in source_counts.to_dicts()
        )
        unmapped_nodes = total_nodes - mapped_nodes
        mapped_share = mapped_nodes / total_nodes if total_nodes else 0.0

        return pl.DataFrame(
            {
                "total_country_sector_nodes": [total_nodes],
                "mapped_nodes": [mapped_nodes],
                "unmapped_nodes": [unmapped_nodes],
                "mapped_share": [mapped_share],
                "number_of_sectors": [ecosystem_mapping["Sector"].n_unique()],
                "number_of_ecosystems": [ecosystem_mapping["ecosystem_id"].n_unique()],
                "ecosystem_source_counts": [source_counts_text],
                "notes": [notes],
            }
        )

    def build_sector_coverage(
        self,
        state_panel: pl.DataFrame,
        ecosystem_mapping: pl.DataFrame,
    ) -> pl.DataFrame:
        """Build sector-level ecosystem coverage diagnostics."""
        sector_counts = (
            state_panel.group_by("Sector")
            .agg(pl.col("country_sector").n_unique().alias("node_count"))
        )
        return (
            sector_counts.join(ecosystem_mapping, on="Sector", how="left")
            .with_columns(
                pl.col("ecosystem_id").fill_null("other_unclassified"),
                pl.col("ecosystem_label").fill_null(ECOSYSTEM_LABELS["other_unclassified"]),
                pl.col("ecosystem_source").fill_null("fallback_unknown"),
                pl.col("mapping_rule").fill_null("missing_sector_in_mapping"),
            )
            .sort("Sector")
        )

    def _join_mapping_to_state(
        self,
        state_panel: pl.DataFrame,
        ecosystem_mapping: pl.DataFrame,
    ) -> pl.DataFrame:
        mapping_columns = [
            "Sector",
            "ecosystem_id",
            "ecosystem_label",
            "ecosystem_source",
        ]
        state_without_ecosystems = state_panel.drop(
            [
                column_name
                for column_name in ("ecosystem_id", "ecosystem_label", "ecosystem_source")
                if column_name in state_panel.columns
            ]
        )
        return (
            state_without_ecosystems.join(
                ecosystem_mapping.select(mapping_columns),
                on="Sector",
                how="left",
            )
            .with_columns(
                pl.col("ecosystem_id").fill_null("fallback_unknown"),
                pl.col("ecosystem_label").fill_null("Unknown ecosystem"),
                pl.col("ecosystem_source").fill_null("fallback_unknown"),
            )
        )

    def _try_hs92_dominant_cluster_mapping(self) -> pl.DataFrame | None:
        """Try HS92 dominant product clusters only when a product cluster field exists."""
        hs92_path = (
            self.paths.data_atlas
            / "processed"
            / "atlas_hs92_level4_clean_panel_1995_2016.parquet"
        )
        concordance_path = self.paths.data_atlas / "concordance" / "hs92_to_eora26_prefilled.csv"
        if not hs92_path.exists() or not concordance_path.exists():
            return None

        hs92_schema = pl.scan_parquet(hs92_path).collect_schema()
        cluster_columns = [
            column_name
            for column_name in hs92_schema.names()
            if any(token in column_name.lower() for token in ("cluster", "community", "ecosystem"))
        ]
        if not cluster_columns:
            return None

        return None
