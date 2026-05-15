from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from src.abm_v4.config import ABMV4Config
from src.abm_v4.emissions import (
    EID_DIAGNOSTIC_CANDIDATE_ID,
    EID_DIAGNOSTIC_D_MIN,
    EID_DIAGNOSTIC_LAMBDA,
    HISTORICAL_FRONTIER_GAP_EID_DIAGNOSTIC_MODE,
    HISTORICAL_FRONTIER_GAP_ONLY_MODE,
    load_eid_diagnostic_scores,
    load_historical_frontier_gap_parameters,
)
from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.state import StateSourceDiagnostic, discover_state_source
from src.abm_v4.validation import (
    OneStepBaseValidationResult,
    build_one_step_base_validation_report,
    missing_one_step_component_paths,
    write_one_step_base_validation_outputs,
)


@dataclass(frozen=True)
class SimulationReadinessReport:
    """Phase 1 readiness report for ABM v4 simulation inputs."""

    state_source: StateSourceDiagnostic
    can_run_base_model: bool


def inspect_base_model_readiness(
    paths: ABMV4Paths,
    config: ABMV4Config,
) -> SimulationReadinessReport:
    """Inspect whether local inputs exist for a future base-model run."""
    state_source = discover_state_source(paths, config.start_year, config.end_year)
    return SimulationReadinessReport(
        state_source=state_source,
        can_run_base_model=state_source.has_source,
    )


@dataclass(frozen=True)
class OneStepBaseRunResult:
    """Result of the one-step base orchestration check."""

    validation: OneStepBaseValidationResult
    reused_existing_outputs: bool
    raw_t_rebuild_skipped: bool


@dataclass(frozen=True)
class MultiYearBaseSimulationResult:
    """Outputs from the historical multi-year ABM v4 base simulation."""

    state_panel: pl.DataFrame
    summary_panel: pl.DataFrame
    validation_report: pl.DataFrame
    yearly_diagnostics: pl.DataFrame


def run_one_step_base_orchestration(
    paths: ABMV4Paths,
    config: ABMV4Config,
    *,
    reuse_existing: bool = True,
    force_rebuild_raw_t_edges: bool = False,
    write_outputs: bool = False,
) -> OneStepBaseRunResult:
    """Run the one-step base integration check from existing component outputs.

    This orchestration layer intentionally does not launch expensive component
    builders by default. It validates the outputs from Phases 2-7B and writes
    only consolidated validation artifacts when requested.
    """
    missing = missing_one_step_component_paths(paths, config)
    if missing:
        missing_lines = "\n".join(
            f"- {name}: {path}" for name, path in sorted(missing.items())
        )
        raw_t_hint = ""
        if (
            "raw_t_supplier_edges" in missing
            or "raw_t_supplier_edge_report" in missing
        ) and not force_rebuild_raw_t_edges:
            raw_t_hint = (
                "\nRaw T edges are expensive and were not rebuilt. Run "
                "`python scripts/run_abm_v4_base.py --build-raw-t-supplier-edges "
                "--create-output-dirs` first, or pass --force-rebuild-raw-t-edges "
                "after implementing/accepting that rebuild."
            )
        raise FileNotFoundError(
            "Cannot run one-step ABM v4 base validation because required component "
            f"outputs are missing:\n{missing_lines}{raw_t_hint}"
        )

    validation = build_one_step_base_validation_report(paths, config)
    if write_outputs:
        write_one_step_base_validation_outputs(paths, validation)
    return OneStepBaseRunResult(
        validation=validation,
        reused_existing_outputs=reuse_existing,
        raw_t_rebuild_skipped=not force_rebuild_raw_t_edges,
    )


class MultiYearBaseSimulator:
    """Conservative historical multi-year ABM v4 base integration loop."""

    def __init__(
        self,
        paths: ABMV4Paths,
        config: ABMV4Config,
        *,
        historical_production_forcing: bool = True,
        reuse_existing: bool = True,
        emissions_transition_mode: str | None = None,
        emissions_parameter_file: Path | str | None = None,
    ) -> None:
        self.paths = paths
        self.config = config
        self.historical_production_forcing = historical_production_forcing
        self.reuse_existing = reuse_existing
        self.emissions_transition_mode = (
            emissions_transition_mode or config.emissions.emissions_transition_mode
        )
        self.emissions_parameter_file = emissions_parameter_file
        self.historical_gap_parameters = load_historical_frontier_gap_parameters(
            emissions_parameter_file
        )
        self._state_for_frontiers: pl.DataFrame | None = None
        self._eid_scores: pl.DataFrame | None = None

    def load_state_panel(self) -> pl.DataFrame:
        """Load the ABM v4 state panel."""
        state_path = self.paths.state_panel_path(self.config.start_year, self.config.end_year)
        if not state_path.exists():
            # Allow smoke windows to reuse the canonical full historical state panel.
            canonical = self.paths.state_panel_path(1995, 2016)
            if canonical.exists():
                return pl.read_parquet(canonical)
            raise FileNotFoundError(f"ABM v4 state panel not found: {state_path}")
        return pl.read_parquet(state_path)

    def load_rewiring_flags(self) -> pl.DataFrame:
        """Load one-step rewiring flags if available."""
        if not self.paths.supplier_rewiring_flags_path.exists():
            return pl.DataFrame(
                {
                    "buyer_country_sector": [],
                    "p_rewire": [],
                    "rewire_flag": [],
                }
            )
        return pl.read_parquet(self.paths.supplier_rewiring_flags_path)

    def load_rewiring_report(self) -> dict[str, object]:
        """Load supplier rewiring diagnostics if available."""
        if not self.paths.supplier_rewiring_report_path.exists():
            return {}
        report = pl.read_csv(self.paths.supplier_rewiring_report_path)
        return report.to_dicts()[0] if not report.is_empty() else {}

    def run(self) -> MultiYearBaseSimulationResult:
        """Run the historical base simulation over the configured year window."""
        loaded_state = self.load_state_panel()
        self._state_for_frontiers = loaded_state
        state = loaded_state.filter(
            pl.col("Year").is_between(self.config.start_year, self.config.end_year)
        )
        if state.is_empty():
            raise ValueError(
                f"No ABM v4 state rows for {self.config.start_year}-{self.config.end_year}."
            )
        required = {"country_sector", "Year", "X_observed", "EI", "Country", "Sector"}
        missing = sorted(required - set(state.columns))
        if missing:
            raise ValueError(f"State panel is missing required simulation columns: {missing}")

        years = sorted(state["Year"].unique().to_list())
        flags = self.load_rewiring_flags()
        rewiring_report = self.load_rewiring_report()
        previous_sim: pl.DataFrame | None = None
        simulated_years: list[pl.DataFrame] = []
        yearly_rows: list[dict[str, object]] = []

        for year in years:
            observed = state.filter(pl.col("Year") == year).unique(subset=["country_sector"])
            if previous_sim is None:
                current = self._initialize_year(observed, flags)
            else:
                current = self._simulate_next_year(
                    observed_next=observed,
                    previous_sim=previous_sim,
                    rewiring_flags=flags,
                )
            simulated_years.append(current)
            yearly_rows.append(self._build_year_summary(current, rewiring_report))
            previous_sim = current

        simulation_panel = pl.concat(simulated_years, how="diagonal_relaxed")
        summary = pl.DataFrame(yearly_rows).sort("year")
        validation = self._build_validation_report(simulation_panel, summary, rewiring_report)
        diagnostics = summary.select(
            "year",
            "mean_rEI_used",
            "median_rEI_used",
            "mean_readiness",
            "mean_ei_gap",
            "rewired_buyer_share",
            "max_supplier_weight_sum_error",
            "bad_transition_flag",
            "status",
            "warnings",
        )
        return MultiYearBaseSimulationResult(
            state_panel=simulation_panel,
            summary_panel=summary,
            validation_report=validation,
            yearly_diagnostics=diagnostics,
        )

    def write_outputs(self, result: MultiYearBaseSimulationResult) -> None:
        """Write multi-year base simulation outputs."""
        self.paths.simulations.mkdir(parents=True, exist_ok=True)
        self.paths.diagnostics.mkdir(parents=True, exist_ok=True)
        self.paths.validation.mkdir(parents=True, exist_ok=True)
        if self.emissions_transition_mode == HISTORICAL_FRONTIER_GAP_ONLY_MODE:
            result.state_panel.write_parquet(
                self.paths.base_multiyear_state_panel_historical_frontier_gap_path
            )
            result.summary_panel.write_csv(
                self.paths.base_multiyear_summary_panel_historical_frontier_gap_path
            )
            result.validation_report.write_csv(
                self.paths.base_multiyear_validation_report_historical_frontier_gap_csv_path
            )
            result.yearly_diagnostics.write_csv(
                self.paths.base_multiyear_yearly_diagnostics_historical_frontier_gap_path
            )
            self.paths.base_multiyear_validation_report_historical_frontier_gap_md_path.write_text(
                self._format_historical_frontier_gap_validation_markdown(result),
                encoding="utf-8",
            )
            return
        if self.emissions_transition_mode == HISTORICAL_FRONTIER_GAP_EID_DIAGNOSTIC_MODE:
            result.state_panel.write_parquet(
                self.paths.base_multiyear_state_panel_EID_diagnostic_path
            )
            result.summary_panel.write_csv(
                self.paths.base_multiyear_summary_panel_EID_diagnostic_path
            )
            result.validation_report.write_csv(
                self.paths.base_multiyear_EID_diagnostic_validation_report_path
            )
            result.yearly_diagnostics.write_csv(
                self.paths.base_multiyear_EID_diagnostic_yearly_diagnostics_path
            )
            return
        result.state_panel.write_parquet(self.paths.base_multiyear_state_panel_path)
        result.summary_panel.write_csv(self.paths.base_multiyear_summary_panel_path)
        result.validation_report.write_csv(self.paths.base_multiyear_validation_report_path)
        result.yearly_diagnostics.write_csv(self.paths.base_multiyear_yearly_diagnostics_path)

    def _initialize_year(self, observed: pl.DataFrame, flags: pl.DataFrame) -> pl.DataFrame:
        return self._base_columns(observed, flags).with_columns(
            pl.col("X_observed").alias("X_sim"),
            pl.col("EI").alias("EI_sim"),
            (pl.col("X_observed") * pl.col("EI")).alias("emissions_sim"),
            pl.lit(1.0).alias("input_feasibility"),
            pl.lit(1.0).alias("production_feasibility_ratio"),
            pl.lit(0.0).alias("rEI_used"),
            pl.lit(0.0).alias("ei_gap"),
            pl.lit(None, dtype=pl.Float64).alias("readiness"),
            pl.lit(self.emissions_transition_mode).alias("emissions_transition_mode"),
            pl.lit(None, dtype=pl.Float64).alias("historical_rho_gap"),
            pl.lit(None, dtype=pl.Float64).alias("historical_tau_gap"),
            pl.lit("").alias("historical_parameter_source"),
            pl.lit(None, dtype=pl.Utf8).alias("EID_score_name"),
            pl.lit(None, dtype=pl.Float64).alias("EID_norm"),
            pl.lit(1.0).alias("D_EID"),
            pl.lit(False).alias("EID_missing_flag"),
            pl.lit(False).alias("EID_fallback_flag"),
            pl.lit(None, dtype=pl.Utf8).alias("EID_candidate_id"),
            pl.lit(None, dtype=pl.Float64).alias("EID_lambda"),
            pl.lit(None, dtype=pl.Float64).alias("EID_d_min"),
            pl.lit(False).alias("bad_transition_node_flag"),
            pl.lit(False).alias("production_constraint_flag"),
            pl.lit(0.0).alias("emissions_decomposition_residual_node"),
        )

    def _simulate_next_year(
        self,
        observed_next: pl.DataFrame,
        previous_sim: pl.DataFrame,
        rewiring_flags: pl.DataFrame,
    ) -> pl.DataFrame:
        base = self._base_columns(observed_next, rewiring_flags)
        previous = previous_sim.select(
            "country_sector",
            pl.col("X_sim").alias("_X_prev"),
            pl.col("EI_sim").alias("_EI_prev"),
            pl.col("cap_sim").alias("_cap_prev"),
            pl.col("gcap_sim").alias("_gcap_prev"),
        )
        panel = base.join(previous, on="country_sector", how="left").with_columns(
            pl.coalesce(["_EI_prev", "EI"]).alias("_EI_base"),
            pl.coalesce(["_cap_prev", "cap_sim"]).alias("_cap_base"),
            pl.coalesce(["_gcap_prev", "gcap_sim"]).alias("_gcap_base"),
        )
        panel = self._add_emissions_transition(panel)
        panel = panel.with_columns(
            pl.when(pl.col("_EI_base") > 0)
            .then(pl.col("_EI_base") * (-pl.col("rEI_used")).exp())
            .otherwise(None)
            .alias("EI_sim"),
            pl.when(pl.lit(self.historical_production_forcing))
            .then(pl.col("X_observed"))
            .otherwise(pl.coalesce(["_X_prev", "X_observed"]) * pl.col("input_feasibility"))
            .alias("X_sim"),
        ).with_columns(
            (pl.col("X_sim") * pl.col("EI_sim")).alias("emissions_sim"),
        ).with_columns(
            (
                (pl.col("emissions_sim") - (pl.col("X_sim") * pl.col("EI_sim"))).abs()
            ).alias("emissions_decomposition_residual_node"),
            pl.lit(False).alias("bad_transition_node_flag"),
            (pl.col("input_feasibility") < 0.999999).alias("production_constraint_flag"),
        )
        return panel.drop(["_X_prev", "_EI_prev", "_cap_prev", "_gcap_prev", "_EI_base", "_cap_base", "_gcap_base"])

    def _base_columns(self, observed: pl.DataFrame, flags: pl.DataFrame) -> pl.DataFrame:
        model_cap = (
            pl.col("general_capability_model")
            if "general_capability_model" in observed.columns
            else pl.col("general_capability")
        )
        model_gcap = (
            pl.col("green_capability_model")
            if "green_capability_model" in observed.columns
            else pl.col("green_capability")
        )
        source_cap = (
            pl.col("general_capability_source")
            if "general_capability_source" in observed.columns
            else pl.lit("legacy_or_unavailable")
        )
        source_gcap = (
            pl.col("green_capability_source")
            if "green_capability_source" in observed.columns
            else pl.lit("legacy_or_unavailable")
        )
        ecosystem_id = (
            pl.col("ecosystem_id") if "ecosystem_id" in observed.columns else pl.lit(None)
        )
        ecosystem_label = (
            pl.col("ecosystem_label") if "ecosystem_label" in observed.columns else pl.lit(None)
        )
        selected = observed.select(
            "country_sector",
            pl.col("Year").alias("year"),
            "Country",
            "Sector",
            ecosystem_id.alias("ecosystem_id"),
            ecosystem_label.alias("ecosystem_label"),
            "X_observed",
            pl.col("EI").alias("EI_observed"),
            "EI",
            (
                pl.col("emissions_observed")
                if "emissions_observed" in observed.columns
                else (pl.col("X_observed") * pl.col("EI"))
            ).alias("emissions_observed"),
            model_cap.cast(pl.Float64, strict=False).alias("_cap_raw"),
            model_gcap.cast(pl.Float64, strict=False).alias("_gcap_raw"),
            source_cap.alias("general_capability_source"),
            source_gcap.alias("green_capability_source"),
            (
                pl.col("network_green_exposure")
                if "network_green_exposure" in observed.columns
                else pl.lit(0.0)
            ).alias("network_green_exposure"),
            (
                pl.col("brown_centrality")
                if "brown_centrality" in observed.columns
                else pl.lit(0.0)
            ).alias("brown_centrality"),
        )
        selected = self._normalize_capabilities(selected).with_columns(
            pl.col("_cap_raw").is_null().alias("capability_model_unavailable_flag"),
            (pl.col("EI").is_null() | (pl.col("EI") <= 0)).alias("invalid_EI_flag"),
            pl.lit(1.0).alias("input_feasibility"),
            pl.lit(1.0).alias("production_feasibility_ratio"),
        )
        if flags.is_empty():
            return selected.with_columns(
                pl.lit(False).alias("supplier_rewired_flag"),
                pl.lit(0.0).alias("p_rewire"),
            )
        return selected.join(
            flags.select(
                pl.col("buyer_country_sector").alias("country_sector"),
                "p_rewire",
                pl.col("rewire_flag").cast(pl.Boolean).alias("supplier_rewired_flag"),
            ),
            on="country_sector",
            how="left",
        ).with_columns(
            pl.col("supplier_rewired_flag").fill_null(False),
            pl.col("p_rewire").fill_null(0.0),
        )

    def _normalize_capabilities(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            self._normalize_expr("_cap_raw").alias("cap_sim"),
            self._normalize_expr("_gcap_raw").alias("gcap_sim"),
        )

    def _add_emissions_transition(self, panel: pl.DataFrame) -> pl.DataFrame:
        if self.emissions_transition_mode in {
            HISTORICAL_FRONTIER_GAP_ONLY_MODE,
            HISTORICAL_FRONTIER_GAP_EID_DIAGNOSTIC_MODE,
        }:
            return self._add_historical_frontier_gap_only(panel)
        return self._add_frontier_gap_and_readiness(panel)

    def _add_historical_frontier_gap_only(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Apply Phase 15 sector-background plus rolling p50 frontier-gap transition."""
        frontier = self._rolling_sector_frontier_for_transition_year(int(panel["year"].max()) - 1)
        out = panel.join(frontier, on="Sector", how="left").with_columns(
            pl.when((pl.col("_EI_base") > 0) & (pl.col("_EI_frontier") > 0))
            .then(
                pl.max_horizontal(
                    pl.lit(0.0),
                    pl.col("_EI_base").log() - pl.col("_EI_frontier").log(),
                )
            )
            .otherwise(None)
            .alias("ei_gap")
        )
        out = self._add_rolling_sector_background(out, int(panel["year"].max()) - 1)
        rho_gap = float(self.historical_gap_parameters["rho_gap"])
        tau_gap = float(self.historical_gap_parameters["tau_gap"])
        if self.emissions_transition_mode == HISTORICAL_FRONTIER_GAP_EID_DIAGNOSTIC_MODE:
            out = self._add_eid_diagnostic_scores(out)
            gap_multiplier = pl.col("D_EID")
        else:
            out = out.with_columns(
                pl.lit(None, dtype=pl.Utf8).alias("EID_score_name"),
                pl.lit(None, dtype=pl.Float64).alias("EID_norm"),
                pl.lit(1.0).alias("D_EID"),
                pl.lit(False).alias("EID_missing_flag"),
                pl.lit(False).alias("EID_fallback_flag"),
                pl.lit(None, dtype=pl.Utf8).alias("EID_candidate_id"),
                pl.lit(None, dtype=pl.Float64).alias("EID_lambda"),
                pl.lit(None, dtype=pl.Float64).alias("EID_d_min"),
            )
            gap_multiplier = pl.lit(1.0)
        out = out.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("readiness"),
            pl.when(pl.col("ei_gap").is_null() | (pl.col("ei_gap") <= 0))
            .then(pl.col("sector_background_trend").fill_null(self.config.emissions.sector_background_fallback))
            .otherwise(
                pl.col("sector_background_trend").fill_null(self.config.emissions.sector_background_fallback)
                + gap_multiplier
                * pl.lit(rho_gap)
                * pl.col("ei_gap")
                / (pl.col("ei_gap") + pl.lit(tau_gap))
            )
            .clip(self.config.emissions.rEI_min, self.config.emissions.rEI_max)
            .alias("rEI_used"),
            pl.lit(self.emissions_transition_mode).alias("emissions_transition_mode"),
            pl.lit(rho_gap).alias("historical_rho_gap"),
            pl.lit(tau_gap).alias("historical_tau_gap"),
            pl.lit(str(self.historical_gap_parameters["parameter_source"])).alias(
                "historical_parameter_source"
            ),
        )
        cap_increment = (
            self.config.capability.delta_cap_param
            * (self.config.capability.cap_max - pl.col("_cap_base").fill_null(pl.col("cap_sim")))
            * (
                self.config.capability.k_cap
                * (pl.col("_cap_base").fill_null(pl.col("cap_sim")) - self.config.capability.tau_cap)
            ).map_elements(lambda value: _sigmoid_for_sim(value), return_dtype=pl.Float64)
        )
        gcap_increment = (
            self.config.capability.delta_gcap_param
            * (self.config.capability.gcap_max - pl.col("_gcap_base").fill_null(pl.col("gcap_sim")))
            * (
                self.config.capability.k_gcap
                * (pl.col("_gcap_base").fill_null(pl.col("gcap_sim")) - self.config.capability.tau_gcap)
            ).map_elements(lambda value: _sigmoid_for_sim(value), return_dtype=pl.Float64)
        )
        return out.with_columns(
            (pl.col("_cap_base").fill_null(pl.col("cap_sim")) + cap_increment)
            .clip(0.0, self.config.capability.cap_max)
            .alias("cap_sim"),
            (pl.col("_gcap_base").fill_null(pl.col("gcap_sim")) + gcap_increment)
            .clip(0.0, self.config.capability.gcap_max)
            .alias("gcap_sim"),
        ).drop(["_EI_frontier"])

    def _load_eid_scores(self) -> pl.DataFrame:
        if self._eid_scores is None:
            self._eid_scores = load_eid_diagnostic_scores(self.paths)
        return self._eid_scores

    def _add_eid_diagnostic_scores(self, panel: pl.DataFrame) -> pl.DataFrame:
        scores = self._load_eid_scores()
        return panel.join(scores, on="country_sector", how="left").with_columns(
            pl.lit(EID_DIAGNOSTIC_CANDIDATE_ID).alias("EID_candidate_id"),
            pl.lit(EID_DIAGNOSTIC_LAMBDA).alias("EID_lambda"),
            pl.lit(EID_DIAGNOSTIC_D_MIN).alias("EID_d_min"),
            pl.col("EID_missing_flag").fill_null(True),
            pl.col("EID_score_name").fill_null("structural_dependence_plus_brown_lockin"),
        ).with_columns(
            (pl.col("EID_missing_flag") | pl.col("EID_norm").is_null()).alias("EID_fallback_flag"),
            pl.when(pl.col("EID_norm").is_null())
            .then(pl.lit(1.0))
            .otherwise(
                (pl.lit(1.0) - pl.lit(EID_DIAGNOSTIC_LAMBDA) * pl.col("EID_norm")).clip(
                    EID_DIAGNOSTIC_D_MIN,
                    1.0,
                )
            )
            .alias("D_EID"),
        )

    def _add_frontier_gap_and_readiness(self, panel: pl.DataFrame) -> pl.DataFrame:
        frontier = (
            panel.filter(pl.col("_EI_base") > 0)
            .group_by("Sector")
            .agg(
                pl.col("_EI_base")
                .quantile(self.config.emissions.ei_frontier_quantile)
                .alias("_EI_frontier")
            )
        )
        out = panel.join(frontier, on="Sector", how="left").with_columns(
            pl.when((pl.col("_EI_base") > 0) & (pl.col("_EI_frontier") > 0))
            .then(
                pl.max_horizontal(
                    pl.lit(0.0),
                    pl.col("_EI_base").log() - pl.col("_EI_frontier").log(),
                )
            )
            .otherwise(None)
            .alias("ei_gap")
        ).with_columns(
            (
                pl.lit(self.config.emissions.theta_intercept)
                + self.config.emissions.theta_gcap * pl.col("_gcap_base").fill_null(0.0)
                + self.config.emissions.theta_cap * pl.col("_cap_base").fill_null(0.0)
                + self.config.emissions.theta_network_green
                * pl.col("network_green_exposure").fill_null(0.0)
                - self.config.emissions.theta_brown_centrality
                * pl.col("brown_centrality").fill_null(0.0)
            ).alias("_readiness_linear")
        ).with_columns(
            (
                self.config.emissions.rho_max
                / (1.0 + (-pl.col("_readiness_linear")).exp())
            ).alias("readiness")
        ).with_columns(
            pl.when(pl.col("ei_gap").is_null() | (pl.col("ei_gap") <= 0))
            .then(0.0)
            .otherwise(
                pl.col("readiness")
                * pl.col("ei_gap")
                / (pl.col("ei_gap") + self.config.emissions.tau_gap)
            )
            .alias("_gap_closure")
        ).with_columns(
            pl.col("_gap_closure")
            .clip(self.config.emissions.rEI_min, self.config.emissions.rEI_max)
            .alias("rEI_used")
        )
        cap_increment = (
            self.config.capability.delta_cap_param
            * (self.config.capability.cap_max - pl.col("_cap_base").fill_null(pl.col("cap_sim")))
            * (
                self.config.capability.k_cap
                * (pl.col("_cap_base").fill_null(pl.col("cap_sim")) - self.config.capability.tau_cap)
            ).map_elements(lambda value: _sigmoid_for_sim(value), return_dtype=pl.Float64)
        )
        gcap_increment = (
            self.config.capability.delta_gcap_param
            * (self.config.capability.gcap_max - pl.col("_gcap_base").fill_null(pl.col("gcap_sim")))
            * (
                self.config.capability.k_gcap
                * (pl.col("_gcap_base").fill_null(pl.col("gcap_sim")) - self.config.capability.tau_gcap)
            ).map_elements(lambda value: _sigmoid_for_sim(value), return_dtype=pl.Float64)
        )
        return out.with_columns(
            (pl.col("_cap_base").fill_null(pl.col("cap_sim")) + cap_increment)
            .clip(0.0, self.config.capability.cap_max)
            .alias("cap_sim"),
            (pl.col("_gcap_base").fill_null(pl.col("gcap_sim")) + gcap_increment)
            .clip(0.0, self.config.capability.gcap_max)
            .alias("gcap_sim"),
        ).drop(["_EI_frontier", "_readiness_linear", "_gap_closure"])

    def _rolling_sector_frontier_for_transition_year(self, transition_year: int) -> pl.DataFrame:
        """Return rolling sector p50 frontiers using only observations up to transition_year."""
        if self._state_for_frontiers is None:
            self._state_for_frontiers = self.load_state_panel()
        valid = self._state_for_frontiers.filter(
            (pl.col("Year") <= transition_year) & (pl.col("EI") > 0)
        )
        if valid.is_empty():
            return pl.DataFrame({"Sector": [], "_EI_frontier": []})
        return valid.group_by("Sector").agg(pl.col("EI").quantile(0.50).alias("_EI_frontier"))

    def _add_rolling_sector_background(self, panel: pl.DataFrame, transition_year: int) -> pl.DataFrame:
        """Add sector median observed rEI up to transition_year as historical background."""
        if self._state_for_frontiers is None:
            self._state_for_frontiers = self.load_state_panel()
        observed = (
            self._state_for_frontiers.sort(["country_sector", "Year"])
            .with_columns(
                pl.col("Year").shift(-1).over("country_sector").alias("_next_year"),
                pl.col("EI").shift(-1).over("country_sector").alias("_next_EI"),
            )
            .filter(
                (pl.col("Year") <= transition_year)
                & (pl.col("_next_year") == pl.col("Year") + 1)
                & (pl.col("EI") > 0)
                & (pl.col("_next_EI") > 0)
            )
            .with_columns((pl.col("EI").log() - pl.col("_next_EI").log()).alias("_observed_rEI"))
        )
        global_median = observed["_observed_rEI"].median() if not observed.is_empty() else None
        fallback = (
            self.config.emissions.sector_background_fallback
            if global_median is None
            else float(global_median)
        )
        background = (
            observed.group_by("Sector")
            .agg(pl.col("_observed_rEI").median().clip(-0.03, 0.05).alias("sector_background_trend"))
            if not observed.is_empty()
            else pl.DataFrame({"Sector": [], "sector_background_trend": []})
        )
        return panel.join(background, on="Sector", how="left").with_columns(
            pl.col("sector_background_trend").fill_null(fallback)
        )

    def _build_year_summary(
        self,
        frame: pl.DataFrame,
        rewiring_report: dict[str, object],
    ) -> dict[str, object]:
        valid = frame.filter(~pl.col("invalid_EI_flag"))
        identity_residual = (valid["emissions_sim"] - valid["X_sim"] * valid["EI_sim"]).abs().max()
        total_observed_emissions = valid["emissions_observed"].sum()
        total_sim_emissions = valid["emissions_sim"].sum()
        total_x_observed = frame["X_observed"].sum()
        total_x_sim = frame["X_sim"].sum()
        warnings: list[str] = []
        if self.historical_production_forcing:
            warnings.append("historical_production_forcing")
        if frame["capability_model_unavailable_flag"].sum() / frame.height > 0.1:
            warnings.append("capability_unavailable_share_above_0.1")
        status = "warning" if warnings else "pass"
        return {
            "year": frame["year"].max(),
            "node_count": frame.height,
            "total_X_observed": total_x_observed,
            "total_X_sim": total_x_sim,
            "aggregate_X_error_pct": _pct_error(total_x_sim, total_x_observed),
            "mean_EI_observed": valid["EI_observed"].mean(),
            "mean_EI_sim": valid["EI_sim"].mean(),
            "median_EI_observed": valid["EI_observed"].median(),
            "median_EI_sim": valid["EI_sim"].median(),
            "total_emissions_observed": total_observed_emissions,
            "total_emissions_sim": total_sim_emissions,
            "aggregate_emissions_error_pct": _pct_error(total_sim_emissions, total_observed_emissions),
            "mean_cap_sim": frame["cap_sim"].mean(),
            "mean_gcap_sim": frame["gcap_sim"].mean(),
            "share_capability_unavailable": frame["capability_model_unavailable_flag"].sum() / frame.height,
            "mean_rEI_used": frame["rEI_used"].mean(),
            "median_rEI_used": frame["rEI_used"].median(),
            "mean_readiness": frame["readiness"].mean(),
            "mean_ei_gap": frame["ei_gap"].mean(),
            "rewired_buyer_share": rewiring_report.get("rewired_buyer_share", 0.0),
            "max_supplier_weight_sum_error": rewiring_report.get("max_updated_weight_sum_error", 0.0),
            "aggregate_production_scale_effect": 0.0 if self.historical_production_forcing else None,
            "aggregate_EI_effect": total_sim_emissions - total_observed_emissions,
            "bad_transition_flag": False,
            "emissions_identity_max_error": identity_residual,
            "historical_production_forcing": self.historical_production_forcing,
            "emissions_transition_mode": self.emissions_transition_mode,
            "historical_rho_gap": self.historical_gap_parameters.get("rho_gap"),
            "historical_tau_gap": self.historical_gap_parameters.get("tau_gap"),
            "historical_parameter_source": self.historical_gap_parameters.get("parameter_source"),
            "status": status,
            "warnings": "; ".join(warnings),
        }

    def _build_validation_report(
        self,
        simulation_panel: pl.DataFrame,
        summary: pl.DataFrame,
        rewiring_report: dict[str, object],
    ) -> pl.DataFrame:
        required = {
            "country_sector",
            "year",
            "X_sim",
            "EI_sim",
            "emissions_sim",
            "cap_sim",
            "gcap_sim",
        }
        missing = sorted(required - set(simulation_panel.columns))
        max_identity = summary["emissions_identity_max_error"].max()
        valid_ei_positive = simulation_panel.filter(~pl.col("invalid_EI_flag"))["EI_sim"].min() > 0
        max_weight_error = float(rewiring_report.get("max_updated_weight_sum_error", 0.0) or 0.0)
        blocking: list[str] = []
        warnings: list[str] = []
        if missing:
            blocking.append(f"missing required columns: {', '.join(missing)}")
        if max_identity is not None and max_identity > 1e-6:
            blocking.append(f"emissions identity max error {max_identity}")
        if not valid_ei_positive:
            blocking.append("non-positive simulated EI for valid nodes")
        if max_weight_error > 1e-8:
            blocking.append(f"supplier weight sum error {max_weight_error}")
        if self.historical_production_forcing:
            warnings.append("production is historically forced")
        if summary["aggregate_emissions_error_pct"].abs().max() > 0.5:
            warnings.append("high aggregate emissions error")
        status = "fail" if blocking else ("warning" if warnings else "pass")
        return pl.DataFrame(
            {
                "simulation_start_year": [self.config.start_year],
                "simulation_end_year": [self.config.end_year],
                "year_count": [summary.height],
                "historical_production_forcing": [self.historical_production_forcing],
                "emissions_transition_mode": [self.emissions_transition_mode],
                "historical_parameter_source": [
                    self.historical_gap_parameters.get("parameter_source", "")
                ],
                "status": [status],
                "missing_required_columns": ["; ".join(missing)],
                "max_emissions_identity_error": [max_identity],
                "valid_ei_positive": [valid_ei_positive],
                "max_supplier_weight_sum_error": [max_weight_error],
                "warnings": ["; ".join(warnings)],
                "blocking_issues": ["; ".join(blocking)],
                "raw_t_rebuilt": [False],
                "scenario_outputs_created": [False],
            }
        )

    def _format_historical_frontier_gap_validation_markdown(
        self,
        result: MultiYearBaseSimulationResult,
    ) -> str:
        """Render a compact validation report for the calibrated-historical run."""
        validation = result.validation_report.to_dicts()[0]
        latest = result.summary_panel.sort("year").tail(1).to_dicts()[0]
        return "\n".join(
            [
                "# ABM v4 Historical Frontier-Gap Multi-Year Validation",
                "",
                "This run uses `historical_frontier_gap_only`; it is not a scenario.",
                "",
                "## Summary",
                "",
                f"- Status: {validation['status']}",
                f"- Years: {validation['simulation_start_year']}-{validation['simulation_end_year']}",
                f"- Historical production forcing: {validation['historical_production_forcing']}",
                f"- Parameter source: {validation['historical_parameter_source']}",
                f"- Latest aggregate emissions pct error: {latest['aggregate_emissions_error_pct']}",
                f"- Mean rEI used latest year: {latest['mean_rEI_used']}",
                f"- Max emissions identity error: {validation['max_emissions_identity_error']}",
                "",
                "## Caveat",
                "",
                "Production remains historically forced and this run is not scenario-ready.",
            ]
        ) + "\n"

    def _normalize_expr(self, column_name: str) -> pl.Expr:
        value = pl.col(column_name).cast(pl.Float64, strict=False)
        min_value = value.min()
        max_value = value.max()
        return (
            pl.when(value.is_null())
            .then(None)
            .when((min_value >= 0.0) & (max_value <= 1.0))
            .then(value)
            .when(max_value > min_value)
            .then((value - min_value) / (max_value - min_value))
            .otherwise(0.0)
        )


def _sigmoid_for_sim(value: float | None) -> float:
    if value is None:
        return 0.0
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _pct_error(simulated: float | None, observed: float | None) -> float | None:
    if observed is None or observed == 0 or simulated is None:
        return None
    return (simulated - observed) / observed
