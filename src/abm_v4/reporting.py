from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import polars as pl

from src.abm_v4.paths import ABMV4Paths


FINAL_TABLE_NAMES = (
    "final_two_rule_summary.csv",
    "final_mechanism_status.csv",
    "final_scenario_blockers.csv",
    "final_abm_v5_priorities.csv",
    "final_portfolio_metrics.csv",
    "final_report_table_index.csv",
)

FINAL_PLOT_NAMES = (
    "abm_v4_two_rule_tradeoff",
    "abm_v4_validation_objective_matrix",
    "abm_v4_mechanism_funnel",
    "abm_v4_scenario_readiness_blockers",
    "abm_v4_abm_v5_research_priorities",
    "abm_v4_portfolio_story_map",
    "abm_v4_hypothesis_status",
)

NARRATIVE_PLOT_NAMES = (
    "abm_v4_architecture_layers",
    "abm_v4_emissions_decomposition_logic",
    "abm_v4_two_rule_metric_tradeoff",
    "abm_v4_mechanism_decision_tree",
    "abm_v4_capability_source_coverage",
    "abm_v4_q_energy_mix_quality_boundary",
    "abm_v4_china_electricity_boundary_case",
    "abm_v4_scenario_readiness_checklist",
    "abm_v4_to_v5_roadmap",
    "abm_v4_hypothesis_status_table",
)

NARRATIVE_SOURCE_TABLE_NAMES = (
    "architecture_layers_source.csv",
    "emissions_decomposition_logic_source.csv",
    "two_rule_metric_tradeoff_source.csv",
    "mechanism_decision_tree_source.csv",
    "capability_source_coverage_source.csv",
    "q_energy_mix_quality_boundary_source.csv",
    "china_electricity_boundary_case_source.csv",
    "scenario_readiness_checklist_source.csv",
    "abm_v4_to_v5_roadmap_source.csv",
    "hypothesis_status_table_source.csv",
)

POLISHED_PLOT_NAMES = (
    "abm_v4_architecture_layers_polished",
    "abm_v4_emissions_decomposition_logic_polished",
    "abm_v4_two_rule_scorecard",
    "abm_v4_mechanism_status_grid",
    "abm_v4_capability_source_coverage_polished",
    "abm_v4_q_energy_mix_quality_boundary_polished",
    "abm_v4_q_energy_mix_quality_boundary_web",
    "abm_v4_china_electricity_boundary_case_polished",
    "abm_v4_scenario_readiness_checklist_polished",
    "abm_v4_to_v5_roadmap_polished",
)

POLISHED_SOURCE_TABLE_NAMES = (
    "architecture_layers_polished_source.csv",
    "emissions_decomposition_logic_polished_source.csv",
    "two_rule_scorecard_source.csv",
    "mechanism_status_grid_source.csv",
    "capability_source_coverage_polished_source.csv",
    "q_energy_mix_quality_boundary_polished_source.csv",
    "china_electricity_boundary_case_polished_source.csv",
    "scenario_readiness_checklist_polished_source.csv",
    "abm_v4_to_v5_roadmap_polished_source.csv",
    "hypothesis_status_report_table.csv",
    "final_visual_selection_manifest.csv",
)


@dataclass(frozen=True)
class ABMV4FinalArtifactResult:
    """Paths written by the final ABM v4 artifact builder."""

    table_paths: tuple[Path, ...]
    plot_paths: tuple[Path, ...]
    copied_plot_paths: tuple[Path, ...]
    artifact_index_path: Path


class ABMV4FinalArtifactBuilder:
    """Build final ABM v4 tables, plots, and artifact index from Phase 28 outputs."""

    def __init__(self, paths: ABMV4Paths) -> None:
        self.paths = paths

    def run(self, *, write_outputs: bool, copy_portfolio_plots: bool = True) -> ABMV4FinalArtifactResult:
        """Build final artifacts, writing only when explicitly requested."""
        self.validate_required_phase28_outputs()
        tables = self.build_final_tables()
        plots = self.build_final_plots()
        if not write_outputs:
            return ABMV4FinalArtifactResult((), (), (), self.paths.final_artifact_index_path)

        self.paths.ensure_final_artifact_directories()
        table_paths = self.write_tables(tables)
        plot_paths = self.write_plots(plots, self.paths.final_plots)
        copied_paths: tuple[Path, ...] = ()
        if copy_portfolio_plots:
            copied_paths = self.copy_plots_to_portfolio_directory(plot_paths)
        artifact_index = self.build_artifact_index(table_paths, plot_paths, copied_paths)
        artifact_index.write_csv(self.paths.final_artifact_index_path)
        return ABMV4FinalArtifactResult(
            table_paths=table_paths,
            plot_paths=plot_paths,
            copied_plot_paths=copied_paths,
            artifact_index_path=self.paths.final_artifact_index_path,
        )

    def validate_required_phase28_outputs(self) -> None:
        """Fail clearly if Phase 28 finalization artifacts are missing."""
        missing = [path for path in self.required_phase28_paths() if not path.exists()]
        if missing:
            formatted = "\n".join(f"- {self._relative(path)}" for path in missing)
            raise FileNotFoundError(
                "Cannot build ABM v4 final plots/tables because required Phase 28 outputs are missing.\n"
                f"{formatted}\n"
                "Run: python scripts/run_abm_v4_base.py --finalize-abm-v4 --create-output-dirs"
            )

    def required_phase28_paths(self) -> tuple[Path, ...]:
        """Return the required Phase 28 validation outputs."""
        return (
            self.paths.final_abm_v4_consolidation_report_path,
            self.paths.final_abm_v4_hypothesis_status_path,
            self.paths.final_model_boundary_statement_path,
            self.paths.final_rejected_mechanism_register_path,
            self.paths.final_scenario_readiness_assessment_path,
            self.paths.final_surviving_rule_comparison_path,
            self.paths.final_validation_objective_matrix_path,
            self.paths.final_abm_v5_research_agenda_path,
            self.paths.final_abm_v4_portfolio_summary_path,
        )

    def optional_phase28_paths(self) -> tuple[Path, ...]:
        """Return optional diagnostic source files used when available."""
        return (
            self.paths.multiyear_base_model_comparison_csv_path,
            self.paths.multiyear_error_summary_path,
            self.paths.multiyear_error_by_sector_path,
            self.paths.multiyear_error_by_country_path,
            self.paths.multiyear_EID_diagnostic_comparison_path,
            self.paths.adaptive_EID_model_comparison_path,
            self.paths.q_energy_mix_hypothesis_tests_path,
            self.paths.q_energy_mix_recommendation_path,
            self.paths.q_energy_mix_china_electricity_audit_path,
            self.paths.q_energy_mix_predictor_screening_path,
            self.paths.transition_rule_aggregate_contribution_path,
            self.paths.electricity_transition_regime_recommendation_path,
        )

    def build_final_tables(self) -> dict[str, pl.DataFrame]:
        """Build all final clean tables in memory."""
        tables: dict[str, pl.DataFrame] = {}
        tables["final_two_rule_summary.csv"] = self.build_two_rule_summary()
        tables["final_mechanism_status.csv"] = self.build_mechanism_status()
        tables["final_scenario_blockers.csv"] = self.build_scenario_blockers()
        tables["final_abm_v5_priorities.csv"] = self.build_abm_v5_priorities()
        tables["final_portfolio_metrics.csv"] = self.build_portfolio_metrics()
        tables["final_report_table_index.csv"] = self.build_report_table_index(
            tuple(self.paths.final_tables / name for name in tables)
        )
        return tables

    def build_two_rule_summary(self) -> pl.DataFrame:
        """Build the final two-rule summary table."""
        source = pl.read_csv(self.paths.final_surviving_rule_comparison_path)
        rows = []
        for rule_name in ("frontier_gap_readiness", "historical_frontier_gap_only"):
            row = source.filter(pl.col("rule_name") == rule_name).row(0, named=True)
            rows.append(
                {
                    "rule_name": rule_name,
                    "retained_as": row.get("retained_as", ""),
                    "theoretical_role": row.get("theoretical_role", ""),
                    "strengths": row.get("strengths", ""),
                    "weaknesses": row.get("weaknesses", ""),
                    "scenario_status": row.get("scenario_use_status", "not_scenario_ready"),
                    "final_interpretation": self._two_rule_interpretation(rule_name),
                }
            )
        return pl.DataFrame(rows)

    def build_mechanism_status(self) -> pl.DataFrame:
        """Build final mechanism status, preserving rejected mechanisms as evidence where appropriate."""
        rejected = pl.read_csv(self.paths.final_rejected_mechanism_register_path)
        hypotheses = pl.read_csv(self.paths.final_abm_v4_hypothesis_status_path)
        rows = [
            self._surviving_mechanism_row(
                "frontier_gap_readiness",
                "retained",
                "Phase 28 retained aggregate-safe baseline.",
                "historical aggregate-safety baseline",
                "ABM v4 historical diagnostic baseline",
                "not_scenario_ready",
            ),
            self._surviving_mechanism_row(
                "historical_frontier_gap_only",
                "retained",
                "Phase 28 retained transition-mechanism benchmark.",
                "frontier-gap transition benchmark",
                "ABM v4 mechanism benchmark",
                "not_scenario_ready",
            ),
        ]
        rows.extend(
            [
                self._rejected_mechanism_row(rejected, "legacy_raw_log emissions rule", "legacy_raw_log"),
                self._rejected_mechanism_row(
                    rejected,
                    "fixed EID dampener",
                    "fixed_EID_dampener",
                    retained_value="ontology evidence; rejected as ABM v4 transition rule",
                ),
                self._rejected_mechanism_row(
                    rejected,
                    "adaptive EID dampener",
                    "adaptive_EID_dampener",
                    retained_value="ontology evidence and calibration warning; rejected as ABM v4 transition rule",
                ),
                self._rejected_mechanism_row(
                    rejected,
                    "EID diagnostic multi-year mode",
                    "EID_multi_year_diagnostic",
                    retained_value="ontology evidence and failure-mode diagnostic evidence",
                ),
                self._rejected_mechanism_row(
                    rejected,
                    "Q energy mix country-sector transition rule",
                    "Q_energy_mix_country_sector_rule",
                    retained_value="aggregate diagnostic and ABM v5 evidence; rejected as ABM v4 node-level rule",
                ),
                self._surviving_mechanism_row(
                    "Q_energy_mix_aggregate_diagnostic",
                    "diagnostic_only",
                    self._hypothesis_evidence(hypotheses, "Q_energy_mix_as_aggregate_diagnostic"),
                    "aggregate diagnostic evidence and ABM v5 fuel-structure motivation",
                    "ABM v5 energy/fuel structure design input",
                    "not_scenario_ready",
                ),
                self._rejected_mechanism_row(
                    rejected,
                    "historical residual as scenario-facing rule",
                    "historical_residual",
                ),
                self._rejected_mechanism_row(
                    rejected,
                    "electricity-specific transition patch",
                    "electricity_specific_patch",
                ),
            ]
        )
        return pl.DataFrame(rows)

    def build_scenario_blockers(self) -> pl.DataFrame:
        """Build final scenario-readiness blocker table."""
        readiness = pl.read_csv(self.paths.final_scenario_readiness_assessment_path)
        boundary = self.paths.final_model_boundary_statement_path.read_text(encoding="utf-8")
        rows = [
            self._blocker_row(
                "historical production forcing",
                "historical production forcing remains central" if "historical production forcing" in boundary.lower() else self._readiness_evidence(readiness, "production_dynamics"),
                "Scenario production paths would otherwise be anchored to observed history.",
                "Build endogenous demand, capacity, substitution, and production propagation.",
            ),
            self._blocker_row("no single scenario-facing emissions rule", self._readiness_evidence(readiness, "emissions_transition_rule"), "Two surviving rules optimize different validation objectives.", "Model fuel and policy mechanisms before selecting a scenario rule."),
            self._blocker_row("missing energy/fuel structure", self._readiness_evidence(readiness, "electricity_energy_system"), "Electricity and high-emissions nodes need fuel-system state variables.", "Use cleaner generation, fuel-use, and capacity data."),
            self._blocker_row("missing policy/institutional variables", self._readiness_evidence(readiness, "policy_institutional_variables"), "Transition accelerations and stalls likely depend on policy and institutions.", "Add policy, investment, carbon-pricing, subsidy, and regulation variables."),
            self._blocker_row("missing capital-stock inertia", "Phase 28 ABM v5 agenda identifies capital-stock inertia as high priority.", "Long-lived assets can slow transition even when frontier gaps suggest improvement.", "Add asset age, capacity, plant, and capital-stock constraints."),
            self._blocker_row("incomplete endogenous production dynamics", self._readiness_evidence(readiness, "production_dynamics"), "Counterfactual output cannot yet emerge from the model itself.", "Replace observed output forcing with validated production-network dynamics."),
            self._blocker_row("data quality limits for Q energy mix", self._readiness_evidence(readiness, "data_quality"), "Country-sector energy-mix data are not clean enough for behavioural rules.", "Reconstruct or source cleaner node-level energy-system data."),
        ]
        return pl.DataFrame(rows)

    def build_abm_v5_priorities(self) -> pl.DataFrame:
        """Build final ABM v5 priority table."""
        agenda = pl.read_csv(self.paths.final_abm_v5_research_agenda_path)
        return agenda.select(
            [
                pl.col("research_priority").alias("priority"),
                "motivation_from_abm_v4",
                "required_data",
                pl.col("candidate_mechanism").alias("mechanism"),
                "candidate_agent_type",
                pl.col("expected_validation_test").alias("validation_test"),
                "priority_level",
            ]
        )

    def build_portfolio_metrics(self) -> pl.DataFrame:
        """Build compact portfolio metrics for final presentation assets."""
        return pl.DataFrame(
            [
                {"metric": "country-sector agents", "value": "4,915", "interpretation": "Country-sector nodes in the ABM v4 historical state panel."},
                {"metric": "historical years", "value": "1995-2016", "interpretation": "Historical validation window."},
                {"metric": "state-panel rows", "value": "108,130", "interpretation": "Country-sector-year rows in the core state panel."},
                {"metric": "final test count", "value": "355 passed", "interpretation": "Final ABM v4 test status at Phase 28 close."},
                {"metric": "surviving rules", "value": "2", "interpretation": "Two final rules retained for distinct historical diagnostic roles."},
                {"metric": "final status", "value": "historical diagnostic framework", "interpretation": "ABM v4 is not a scenario-ready forecasting model."},
                {"metric": "scenario readiness", "value": "not scenario-ready", "interpretation": "Scenarios remain blocked until ABM v5 mechanisms are built."},
            ]
        )

    def build_report_table_index(self, table_paths: tuple[Path, ...]) -> pl.DataFrame:
        """Index final clean tables and intended uses."""
        source_files = {
            "final_two_rule_summary.csv": "final_surviving_rule_comparison.csv",
            "final_mechanism_status.csv": "final_rejected_mechanism_register.csv; final_abm_v4_hypothesis_status.csv",
            "final_scenario_blockers.csv": "final_scenario_readiness_assessment.csv; final_model_boundary_statement.md",
            "final_abm_v5_priorities.csv": "final_abm_v5_research_agenda.csv",
            "final_portfolio_metrics.csv": "Phase 28 final status constants",
            "final_report_table_index.csv": "Phase 29A generated table list",
        }
        intended_use = {
            "final_two_rule_summary.csv": "Report table for the final two-rule interpretation.",
            "final_mechanism_status.csv": "Report table for retained, rejected, and diagnostic mechanisms.",
            "final_scenario_blockers.csv": "Report table explaining why scenarios remain premature.",
            "final_abm_v5_priorities.csv": "Report and portfolio source for ABM v5 research agenda.",
            "final_portfolio_metrics.csv": "Portfolio-ready quantitative summary.",
            "final_report_table_index.csv": "Traceable index of final clean tables.",
        }
        rows = []
        for path in table_paths:
            rows.append(
                {
                    "table_name": path.name,
                    "path": self._relative(path),
                    "source_files": source_files.get(path.name, ""),
                    "intended_use": intended_use.get(path.name, ""),
                }
            )
        return pl.DataFrame(rows)

    def build_final_plots(self) -> dict[str, plt.Figure]:
        """Build all final static plots in memory."""
        return {
            "abm_v4_two_rule_tradeoff": self.plot_two_rule_tradeoff(),
            "abm_v4_validation_objective_matrix": self.plot_validation_objective_matrix(pl.read_csv(self.paths.final_validation_objective_matrix_path)),
            "abm_v4_mechanism_funnel": self.plot_mechanism_funnel(self.build_mechanism_status()),
            "abm_v4_scenario_readiness_blockers": self.plot_scenario_readiness_blockers(pl.read_csv(self.paths.final_scenario_readiness_assessment_path)),
            "abm_v4_abm_v5_research_priorities": self.plot_abm_v5_research_priorities(self.build_abm_v5_priorities()),
            "abm_v4_portfolio_story_map": self.plot_portfolio_story_map(),
            "abm_v4_hypothesis_status": self.plot_hypothesis_status(pl.read_csv(self.paths.final_abm_v4_hypothesis_status_path)),
        }

    def plot_two_rule_tradeoff(self) -> plt.Figure:
        """Plot the final two-rule tradeoff as an ordinal comparison."""
        labels = ["Aggregate safety", "Mechanism clarity"]
        data = {
            "frontier_gap_readiness": [3, 2],
            "historical_frontier_gap_only": [2, 3],
        }
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x_positions = range(len(labels))
        width = 0.35
        ax.bar([x - width / 2 for x in x_positions], data["frontier_gap_readiness"], width, label="frontier_gap_readiness")
        ax.bar([x + width / 2 for x in x_positions], data["historical_frontier_gap_only"], width, label="historical_frontier_gap_only")
        ax.set_xticks(list(x_positions), labels)
        ax.set_ylim(0, 3.5)
        ax.set_ylabel("Ordinal support")
        ax.set_title("ABM v4 Final Two-Rule Tradeoff")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
        ax.text(-0.25, 3.2, "aggregate-safe baseline", fontsize=8)
        ax.text(0.75, 3.2, "transition-mechanism benchmark", fontsize=8)
        fig.tight_layout()
        return fig

    def plot_validation_objective_matrix(self, matrix: pl.DataFrame) -> plt.Figure:
        """Plot validation objective support by evidence branch."""
        objective_col = "objective"
        columns = [
            "frontier_gap_readiness_assessment",
            "historical_frontier_gap_only_assessment",
            "EID_assessment",
            "Q_energy_mix_assessment",
        ]
        display_cols = ["frontier_gap_readiness", "historical_frontier_gap_only", "EID", "Q energy mix"]
        values = [[self._assessment_score(row[col]) for col in columns] for row in matrix.to_dicts()]
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        image = ax.imshow(values, aspect="auto")
        ax.set_xticks(range(len(display_cols)), display_cols, rotation=25, ha="right")
        ax.set_yticks(range(matrix.height), [self._display_label(value) for value in matrix[objective_col].to_list()])
        ax.set_title("ABM v4 Validation Objective Matrix")
        for y, row in enumerate(values):
            for x, value in enumerate(row):
                ax.text(x, y, str(value), ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, shrink=0.75, label="Support score")
        fig.tight_layout()
        return fig

    def plot_mechanism_funnel(self, mechanism_table: pl.DataFrame) -> plt.Figure:
        """Plot the final mechanism-testing funnel."""
        steps = [
            "raw log rule\nrejected",
            "frontier_gap_readiness\naggregate-safe baseline",
            "historical_frontier_gap_only\nbenchmark",
            "EID\nontology evidence",
            "Q energy mix\naggregate/ABM v5 evidence",
            "scenarios\nblocked",
            "ABM v5\nagenda",
        ]
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.axis("off")
        y = 0.55
        xs = [0.06, 0.20, 0.36, 0.51, 0.66, 0.81, 0.94]
        for i, (x, step) in enumerate(zip(xs, steps)):
            ax.text(x, y, step, ha="center", va="center", fontsize=8, bbox={"boxstyle": "round", "pad": 0.35, "fill": False})
            if i < len(xs) - 1:
                ax.annotate("", xy=(xs[i + 1] - 0.055, y), xytext=(x + 0.055, y), arrowprops={"arrowstyle": "->"})
        ax.set_title("ABM v4 Mechanism Funnel")
        fig.tight_layout()
        return fig

    def plot_scenario_readiness_blockers(self, blockers: pl.DataFrame) -> plt.Figure:
        """Plot scenario-readiness blockers by status."""
        status_score = {"blocked": 3, "not_scenario_ready": 3, "limited": 2}
        label_column = "readiness_dimension" if "readiness_dimension" in blockers.columns else "blocker"
        labels = [self._display_label(value) for value in blockers[label_column].to_list()]
        values = [
            status_score.get(row.get("status", "blocked"), 3)
            for row in blockers.to_dicts()
        ]
        fig, ax = plt.subplots(figsize=(9, 5))
        y_positions = range(len(labels))
        ax.barh(list(y_positions), values)
        ax.set_yticks(list(y_positions), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 3.5)
        ax.set_xlabel("Blocker severity")
        ax.set_title("ABM v4 Scenario Readiness Blockers")
        fig.tight_layout()
        return fig

    def plot_abm_v5_research_priorities(self, priorities: pl.DataFrame) -> plt.Figure:
        """Plot the ABM v5 priority stack."""
        labels = [self._display_label(value) for value in priorities["priority"].to_list()]
        values = [3 if value == "high" else 2 for value in priorities["priority_level"].to_list()]
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.barh(range(len(labels)), values)
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 3.5)
        ax.set_xlabel("Priority level")
        ax.set_title("ABM v5 Research Priorities from ABM v4")
        fig.tight_layout()
        return fig

    def plot_portfolio_story_map(self) -> plt.Figure:
        """Plot a compact portfolio story map."""
        steps = [
            "theory",
            "model\narchitecture",
            "historical\nvalidation",
            "failed\nmechanisms",
            "model\nboundary",
            "ABM v5\nagenda",
        ]
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.axis("off")
        xs = [0.08, 0.24, 0.40, 0.56, 0.72, 0.88]
        for i, (x, step) in enumerate(zip(xs, steps)):
            ax.text(x, 0.55, step, ha="center", va="center", fontsize=9, bbox={"boxstyle": "round", "pad": 0.35, "fill": False})
            if i < len(xs) - 1:
                ax.annotate("", xy=(xs[i + 1] - 0.055, 0.55), xytext=(x + 0.055, 0.55), arrowprops={"arrowstyle": "->"})
        ax.set_title("ABM v4 Portfolio Story Map")
        fig.tight_layout()
        return fig

    def plot_hypothesis_status(self, hypotheses: pl.DataFrame) -> plt.Figure:
        """Plot final hypothesis outcomes."""
        counts = hypotheses.group_by("status").len().sort("status")
        labels = [self._display_label(value) for value in counts["status"].to_list()]
        values = counts["len"].to_list()
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.bar(labels, values)
        ax.set_ylabel("Hypotheses")
        ax.set_title("ABM v4 Final Hypothesis Status")
        ax.tick_params(axis="x", rotation=20)
        for row in hypotheses.to_dicts():
            if row["hypothesis"] in {
                "frontier_gap_readiness_aggregate_safe",
                "historical_frontier_gap_transition_benchmark",
                "EID_as_transition_dampener",
                "EID_as_ontology_signal",
                "Q_energy_mix_as_country_sector_rule",
                "Q_energy_mix_as_aggregate_diagnostic",
                "ABM_v4_scenario_ready",
            }:
                ax.text(0.98, 0.95 - 0.07 * len(ax.texts), self._wrap(f"{row['hypothesis']}: {row['status']}", 42), transform=ax.transAxes, ha="right", va="top", fontsize=7)
        fig.tight_layout()
        return fig

    def write_tables(self, tables: dict[str, pl.DataFrame]) -> tuple[Path, ...]:
        """Write final clean tables."""
        paths = []
        for name in FINAL_TABLE_NAMES:
            path = self.paths.final_tables / name
            tables[name].write_csv(path)
            paths.append(path)
        return tuple(paths)

    def write_plots(self, plots: dict[str, plt.Figure], output_dir: Path) -> tuple[Path, ...]:
        """Write final plots as PNG and SVG."""
        paths: list[Path] = []
        for name in FINAL_PLOT_NAMES:
            figure = plots[name]
            for suffix in (".png", ".svg"):
                path = output_dir / f"{name}{suffix}"
                figure.savefig(path, bbox_inches="tight", dpi=180)
                paths.append(path)
            plt.close(figure)
        return tuple(paths)

    def copy_plots_to_portfolio_directory(self, plot_paths: tuple[Path, ...]) -> tuple[Path, ...]:
        """Copy final plots to the portfolio-ready output plot folder."""
        copied_paths = []
        self.paths.outputs_plots_abm_v4_final.mkdir(parents=True, exist_ok=True)
        for path in plot_paths:
            destination = self.paths.outputs_plots_abm_v4_final / path.name
            copy2(path, destination)
            copied_paths.append(destination)
        return tuple(copied_paths)

    def build_artifact_index(
        self,
        table_paths: tuple[Path, ...],
        plot_paths: tuple[Path, ...],
        copied_plot_paths: tuple[Path, ...],
    ) -> pl.DataFrame:
        """Build final artifact index including generated and source artifacts."""
        rows = []
        for path in table_paths:
            rows.append(self._artifact_row("final_table", path, "Final clean ABM v4 table.", "LaTeX report and portfolio source material."))
        for path in plot_paths:
            rows.append(self._artifact_row("final_plot", path, "Final static ABM v4 plot.", "LaTeX report figure source."))
        for path in copied_plot_paths:
            rows.append(self._artifact_row("portfolio_plot_copy", path, "Portfolio-ready copy of final ABM v4 plot.", "Portfolio webpage asset source."))
        for path in (*self.required_phase28_paths(), *self.optional_phase28_paths()):
            if path.exists():
                rows.append(self._artifact_row("source_validation_table", path, "Source validation artifact used by Phase 29A.", "Traceability and audit."))
        return pl.DataFrame(rows)

    def _surviving_mechanism_row(
        self,
        mechanism: str,
        status: str,
        evidence: str,
        retained_value: str,
        future_use: str,
        scenario_status: str,
    ) -> dict[str, str]:
        return {
            "mechanism": mechanism,
            "status": status,
            "evidence": evidence,
            "retained_value": retained_value,
            "future_use": future_use,
            "scenario_status": scenario_status,
        }

    def _rejected_mechanism_row(
        self,
        rejected: pl.DataFrame,
        source_mechanism: str,
        mechanism: str,
        retained_value: str | None = None,
    ) -> dict[str, str]:
        row = rejected.filter(pl.col("mechanism") == source_mechanism).row(0, named=True)
        return {
            "mechanism": mechanism,
            "status": row.get("test_result", ""),
            "evidence": row.get("reason_rejected_or_limited", ""),
            "retained_value": retained_value or row.get("retained_value", ""),
            "future_use": row.get("future_use", ""),
            "scenario_status": row.get("scenario_status", ""),
        }

    def _blocker_row(self, blocker: str, evidence: str, why_it_matters: str, required_work: str) -> dict[str, str]:
        return {
            "blocker": blocker,
            "evidence": evidence,
            "why_it_matters": why_it_matters,
            "required_ABM_v5_work": required_work,
        }

    def _artifact_row(self, artifact_type: str, path: Path, description: str, intended_use: str) -> dict[str, str]:
        return {
            "artifact_type": artifact_type,
            "path": self._relative(path),
            "description": description,
            "intended_use": intended_use,
        }

    def _hypothesis_evidence(self, hypotheses: pl.DataFrame, hypothesis: str) -> str:
        matching = hypotheses.filter(pl.col("hypothesis") == hypothesis)
        if matching.is_empty():
            return ""
        return matching.row(0, named=True).get("evidence", "")

    def _readiness_evidence(self, readiness: pl.DataFrame, dimension: str) -> str:
        matching = readiness.filter(pl.col("readiness_dimension") == dimension)
        if matching.is_empty():
            return ""
        row = matching.row(0, named=True)
        return f"{row.get('evidence', '')}; {row.get('blocking_issue', '')}"

    def _two_rule_interpretation(self, rule_name: str) -> str:
        if rule_name == "frontier_gap_readiness":
            return "Aggregate-safe historical baseline; useful for emissions plausibility but less pure as a transition mechanism."
        return "Transition-mechanism benchmark; cleaner frontier-gap interpretation but weaker aggregate safety."

    def _assessment_score(self, value: str) -> int:
        lower = str(value).lower()
        if "wins" in lower or "best" in lower or "supported" in lower or "survives" in lower:
            return 2
        if "partly" in lower or "diagnostic" in lower or "useful" in lower or "aggregate-only" in lower or "concept" in lower:
            return 1
        return 0

    def _display_label(self, value: str) -> str:
        return self._wrap(str(value).replace("_", " "), 28)

    def _wrap(self, value: str, width: int) -> str:
        return "\n".join(textwrap.wrap(value, width=width, break_long_words=False))

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.paths.project_root)).replace("\\", "/")
        except ValueError:
            return str(path).replace("\\", "/")


class ABMV4NarrativePlotBuilder:
    """Forward placeholder replaced later in the module."""

    pass


@dataclass(frozen=True)
class ABMV4PolishedPlotResult:
    """Paths written by the final-polished ABM v4 plot builder."""

    source_table_paths: tuple[Path, ...]
    plot_paths: tuple[Path, ...]
    copied_plot_paths: tuple[Path, ...]
    plot_index_path: Path
    manifest_path: Path


class ABMV4PolishedPlotBuilder(ABMV4NarrativePlotBuilder):
    """Build Phase 29C polished final ABM v4 figures and source tables."""

    def run(self, *, write_outputs: bool, copy_output_plots: bool = True) -> ABMV4PolishedPlotResult:
        """Build polished figures and tables, writing only when explicitly requested."""
        self.validate_required_inputs()
        tables = self.build_source_tables()
        plots = self.build_plots(tables)
        manifest_path = self.paths.final_tables_polished / "final_visual_selection_manifest.csv"
        if not write_outputs:
            for figure in plots.values():
                plt.close(figure)
            return ABMV4PolishedPlotResult((), (), (), self.paths.final_polished_plot_index_path, manifest_path)

        self.paths.ensure_final_polished_artifact_directories()
        table_paths = self.write_source_tables(tables)
        plot_paths = self.write_plots(plots)
        copied_paths: tuple[Path, ...] = ()
        if copy_output_plots:
            copied_paths = self.copy_plots_to_output_directory(plot_paths)
        index = self.build_plot_index(table_paths, plot_paths, copied_paths)
        index.write_csv(self.paths.final_polished_plot_index_path)
        return ABMV4PolishedPlotResult(
            table_paths,
            plot_paths,
            copied_paths,
            self.paths.final_polished_plot_index_path,
            manifest_path,
        )

    def build_source_tables(self) -> dict[str, pl.DataFrame]:
        """Build every polished plot source table in memory."""
        return {
            "architecture_layers_polished_source.csv": self.build_architecture_layers_polished_source(),
            "emissions_decomposition_logic_polished_source.csv": self.build_emissions_decomposition_logic_source(),
            "two_rule_scorecard_source.csv": self.build_two_rule_scorecard_source(),
            "mechanism_status_grid_source.csv": self.build_mechanism_status_grid_source(),
            "capability_source_coverage_polished_source.csv": self.build_capability_source_coverage_source(),
            "q_energy_mix_quality_boundary_polished_source.csv": self.build_q_energy_mix_quality_boundary_polished_source(),
            "china_electricity_boundary_case_polished_source.csv": self.build_china_electricity_boundary_case_polished_source(),
            "scenario_readiness_checklist_polished_source.csv": self.build_scenario_readiness_checklist_polished_source(),
            "abm_v4_to_v5_roadmap_polished_source.csv": self.build_abm_v4_to_v5_roadmap_polished_source(),
            "hypothesis_status_report_table.csv": self.build_hypothesis_status_report_table(),
            "final_visual_selection_manifest.csv": self.build_final_visual_selection_manifest(),
        }

    def build_architecture_layers_polished_source(self) -> pl.DataFrame:
        """Build the shortened top-to-bottom architecture source table."""
        return pl.DataFrame(
            [
                {
                    "layer_order": 1,
                    "layer_name": "Historical data inputs",
                    "component_line": "Eora, emissions, Atlas, IO, Q diagnostics",
                    "metric_line": "Observed 1995-2016 history",
                    "flow_direction": "top_to_bottom",
                },
                {
                    "layer_order": 2,
                    "layer_name": "Country-sector state panel",
                    "component_line": "Nodes with output, EI, capability, ecosystem",
                    "metric_line": "Agent-year validation panel",
                    "flow_direction": "top_to_bottom",
                },
                {
                    "layer_order": 3,
                    "layer_name": "Production network and ecosystems",
                    "component_line": "Supplier edges and opportunity sets",
                    "metric_line": "Networked transition context",
                    "flow_direction": "top_to_bottom",
                },
                {
                    "layer_order": 4,
                    "layer_name": "Behavioural diagnostic layers",
                    "component_line": "Capabilities, suppliers, emissions rules",
                    "metric_line": "Mechanisms tested historically",
                    "flow_direction": "top_to_bottom",
                },
                {
                    "layer_order": 5,
                    "layer_name": "Historical validation and model boundary",
                    "component_line": "Two rules retained; others bounded",
                    "metric_line": "Not scenario-ready",
                    "flow_direction": "top_to_bottom",
                },
            ]
        )

    def build_two_rule_scorecard_source(self) -> pl.DataFrame:
        """Build a qualitative scorecard for the two surviving rules."""
        metrics = self._metrics_by_rule()
        return pl.DataFrame(
            [
                self._scorecard_row(
                    "Aggregate emissions fit",
                    "strong",
                    metrics.get("frontier_gap_readiness", {}).get("aggregate"),
                    "weak",
                    metrics.get("historical_frontier_gap_only", {}).get("aggregate"),
                    "multiyear_base_model_comparison.csv",
                    "Readiness gating improves aggregate emissions plausibility.",
                ),
                self._scorecard_row(
                    "Transition-mechanism fit",
                    "moderate",
                    metrics.get("frontier_gap_readiness", {}).get("rEI"),
                    "strong",
                    metrics.get("historical_frontier_gap_only", {}).get("rEI"),
                    "final_surviving_rule_comparison.csv",
                    "The calibrated historical rule is the cleaner mechanism benchmark.",
                ),
                self._scorecard_row(
                    "Electricity/high-emissions fit",
                    "weak",
                    metrics.get("frontier_gap_readiness", {}).get("electricity"),
                    "weak",
                    metrics.get("historical_frontier_gap_only", {}).get("electricity"),
                    "final_surviving_rule_comparison.csv",
                    "Both rules expose a missing fuel-structure boundary.",
                ),
                self._scorecard_row(
                    "Interpretability",
                    "moderate",
                    "aggregate-safe baseline",
                    "strong",
                    "transition benchmark",
                    "final_surviving_rule_comparison.csv",
                    "ABM v4 retains both because they answer different validation questions.",
                ),
                self._scorecard_row(
                    "Scenario readiness",
                    "blocked",
                    "not scenario-ready",
                    "blocked",
                    "not scenario-ready",
                    "final_scenario_readiness_assessment.csv",
                    "Neither rule should be used as a scenario transition rule.",
                ),
            ]
        )

    def build_mechanism_status_grid_source(self) -> pl.DataFrame:
        """Build short mechanism-status rows for the polished grid."""
        base = self.build_mechanism_decision_tree_source()
        status_map = {
            "retained": "retained ABM v4 rule",
            "rejected as ABM v4 rule": "rejected as ABM v4 rule, retained as evidence",
            "rejected": "rejected",
            "diagnostic only": "diagnostic only",
        }
        rows = []
        for row in base.to_dicts():
            mechanism = str(row["mechanism"])
            status = status_map.get(str(row["status"]), str(row["status"]))
            retained = str(row["retained_value"])
            reason = str(row["reason"])
            if "EID" in mechanism:
                status = "rejected as ABM v4 rule, retained as evidence"
                retained = "ontology evidence"
            if "Q energy" in mechanism:
                status = "rejected as ABM v4 rule, retained as evidence"
                retained = "aggregate and ABM v5 evidence"
                reason = "node-level quality limits"
            rows.append(
                {
                    "mechanism": mechanism,
                    "final_status": status,
                    "retained_value": retained,
                    "main_reason": reason,
                }
            )
        return pl.DataFrame(rows)

    def build_q_energy_mix_quality_boundary_polished_source(self) -> pl.DataFrame:
        """Build a concise Q energy quality boundary source table."""
        base = self.build_q_energy_mix_quality_boundary_source()
        wanted = [
            "all 9 Q energy rows found",
            "broad country-sector coverage",
            "aggregate diagnostic value",
            "China electricity fuel-mix signal",
            "valid node-level shares",
            "no negative values",
            "strong predictive power",
        ]
        return base.filter(pl.col("check").is_in(wanted)).select(
            [
                pl.col("check"),
                pl.col("status"),
                pl.col("quantitative_evidence"),
                pl.col("implication"),
                pl.col("source_file"),
            ]
        )

    def build_q_energy_mix_quality_boundary_web_source(self, table: pl.DataFrame) -> pl.DataFrame:
        """Build the six-row web version of the Q energy boundary table."""
        rows = [
            ("Q rows found", "all 9 Q energy rows found"),
            ("Broad coverage", "broad country-sector coverage"),
            ("Aggregate signal", "aggregate diagnostic value"),
            ("China electricity signal", "China electricity fuel-mix signal"),
            ("Node-level quality", "valid node-level shares"),
            ("Predictive strength", "strong predictive power"),
        ]
        output = []
        for display, source_check in rows:
            match = table.filter(pl.col("check") == source_check)
            row = match.row(0, named=True) if not match.is_empty() else {}
            output.append(
                {
                    "check": display,
                    "status": row.get("status", "partial"),
                    "quantitative_evidence": row.get("quantitative_evidence", "evidence unavailable"),
                    "implication": row.get("implication", ""),
                    "source_file": row.get("source_file", ""),
                }
            )
        return pl.DataFrame(output)

    def build_china_electricity_boundary_case_polished_source(self) -> pl.DataFrame:
        """Build China electricity source data with explicit observed-series labeling."""
        base = self.build_china_electricity_boundary_case_source()
        observed_label = "Observed emissions-intensity reduction (rEI)"
        if self.paths.q_energy_mix_china_electricity_audit_path.exists():
            source = pl.read_csv(self.paths.q_energy_mix_china_electricity_audit_path)
            if "observed_rEI" not in source.columns and "EI_observed" in source.columns:
                observed_label = "Observed emissions intensity"
        return base.with_columns(
            [
                pl.lit(observed_label).alias("observed_series_label"),
                pl.lit("High fossil dependence supports the missing fuel-structure interpretation, but this is not a causal proof.").alias("boundary_caption"),
            ]
        )

    def build_scenario_readiness_checklist_polished_source(self) -> pl.DataFrame:
        """Build a seven-row scenario-readiness checklist."""
        source = pl.read_csv(self.paths.final_scenario_readiness_assessment_path)
        mapping = [
            ("Emissions transition rule", "emissions_transition_rule", "validated transition rule"),
            ("Production dynamics", "production_dynamics", "endogenous production dynamics"),
            ("Electricity and fuel structure", "electricity_energy_system", "fuel and electricity structure"),
            ("Policy variables", "policy_institutional_variables", "policy/institutional variables"),
            ("Capital-stock inertia", "capital_stock_inertia", "capital-stock inertia"),
            ("Data quality", "data_quality", "cleaner rule-grade inputs"),
            ("Overall scenario readiness", "overall_scenario_readiness", "ABM v5 scenario framework"),
        ]
        rows = []
        for label, key, requirement in mapping:
            match = source.filter(pl.col("readiness_dimension") == key)
            row = match.row(0, named=True) if not match.is_empty() else {}
            rows.append(
                {
                    "readiness_dimension": label,
                    "status": row.get("status", "blocked" if key != "overall_scenario_readiness" else "not_scenario_ready"),
                    "blocking_issue": self._shorten(row.get("blocking_issue", "not cleared by final validation"), 58),
                    "abm_v5_requirement": self._shorten(row.get("required_future_work", requirement), 58),
                }
            )
        return pl.DataFrame(rows)

    def build_abm_v4_to_v5_roadmap_polished_source(self) -> pl.DataFrame:
        """Build a shortened ABM v4 to ABM v5 roadmap table."""
        base = self.build_abm_v4_to_v5_roadmap_source()
        rows = []
        for row in base.to_dicts():
            rows.append(
                {
                    "abm_v4_finding": row["abm_v4_finding"],
                    "boundary_identified": self._shorten(row["boundary_identified"], 54),
                    "abm_v5_requirement": row["abm_v5_requirement"],
                }
            )
        return pl.DataFrame(rows)

    def build_hypothesis_status_report_table(self) -> pl.DataFrame:
        """Build the report table for hypotheses without rendering a dense plot."""
        base = self.build_hypothesis_status_table_source()
        return base.rename(
            {
                "short_evidence": "evidence_short",
                "implication": "implication_short",
            }
        ).select(["hypothesis", "status", "evidence_short", "implication_short"])

    def build_final_visual_selection_manifest(self) -> pl.DataFrame:
        """Build final recommendations for report and webpage plot use."""
        rows = [
            ("abm_v4_architecture_layers_polished", True, True, "clear top-to-bottom model architecture", "use as overview"),
            ("abm_v4_emissions_decomposition_logic_polished", True, True, "compact explanation of E and EI logic", "use near methods"),
            ("abm_v4_two_rule_scorecard", True, True, "replaces weak two-point scatter", "use to explain two retained rules"),
            ("abm_v4_mechanism_status_grid", True, False, "documents mechanism testing discipline", "report only if space allows"),
            ("abm_v4_capability_source_coverage_polished", True, False, "shows reduced crude missing-data treatment", "technical report evidence"),
            ("abm_v4_q_energy_mix_quality_boundary_polished", True, False, "explains Q energy boundary with evidence", "report diagnostic"),
            ("abm_v4_q_energy_mix_quality_boundary_web", False, True, "six-row webpage version", "webpage only if space"),
            ("abm_v4_china_electricity_boundary_case_polished", True, True, "boundary case for fuel-mix interpretation", "webpage only if space"),
            ("abm_v4_scenario_readiness_checklist_polished", True, False, "compact scenario blocker checklist", "report conclusion"),
            ("abm_v4_to_v5_roadmap_polished", True, True, "strongest bridge from ABM v4 to ABM v5", "use as closing visual"),
            ("hypothesis_status_report_table.csv", True, False, "table is more readable than dense plot", "do not render full plot"),
            ("Phase 29A plots", False, False, "superseded by narrative and polished visuals", "avoid"),
            ("Phase 29B two-rule scatter", False, False, "replaced by scorecard", "supplementary only"),
            ("Phase 29B mechanism decision tree", False, False, "replaced by status grid", "avoid"),
            ("Phase 29B hypothesis status table", False, False, "too dense as a figure", "use polished table"),
        ]
        return pl.DataFrame(
            rows,
            schema=["plot_file", "use_in_latex_report", "use_in_portfolio_webpage", "reason", "notes"],
            orient="row",
        )

    def build_plots(self, tables: dict[str, pl.DataFrame]) -> dict[str, plt.Figure]:
        """Build every polished plot from source tables."""
        q_web = self.build_q_energy_mix_quality_boundary_web_source(
            tables["q_energy_mix_quality_boundary_polished_source.csv"]
        )
        return {
            "abm_v4_architecture_layers_polished": self.plot_architecture_layers_polished(tables["architecture_layers_polished_source.csv"]),
            "abm_v4_emissions_decomposition_logic_polished": self.plot_emissions_decomposition_logic_polished(tables["emissions_decomposition_logic_polished_source.csv"]),
            "abm_v4_two_rule_scorecard": self.plot_two_rule_scorecard(tables["two_rule_scorecard_source.csv"]),
            "abm_v4_mechanism_status_grid": self.plot_table_grid(tables["mechanism_status_grid_source.csv"], "ABM v4 mechanism status grid", ["mechanism", "final_status", "retained_value", "main_reason"], [0.24, 0.28, 0.22, 0.26]),
            "abm_v4_capability_source_coverage_polished": self.plot_capability_source_coverage_polished(tables["capability_source_coverage_polished_source.csv"]),
            "abm_v4_q_energy_mix_quality_boundary_polished": self.plot_q_energy_mix_quality_boundary_polished(tables["q_energy_mix_quality_boundary_polished_source.csv"], "Q energy mix quality boundary"),
            "abm_v4_q_energy_mix_quality_boundary_web": self.plot_q_energy_mix_quality_boundary_polished(q_web, "Q energy mix boundary: web summary"),
            "abm_v4_china_electricity_boundary_case_polished": self.plot_china_electricity_boundary_case_polished(tables["china_electricity_boundary_case_polished_source.csv"]),
            "abm_v4_scenario_readiness_checklist_polished": self.plot_table_grid(tables["scenario_readiness_checklist_polished_source.csv"], "Scenario readiness checklist", ["readiness_dimension", "status", "blocking_issue", "abm_v5_requirement"], [0.25, 0.16, 0.29, 0.30]),
            "abm_v4_to_v5_roadmap_polished": self.plot_abm_v4_to_v5_roadmap_polished(tables["abm_v4_to_v5_roadmap_polished_source.csv"]),
        }

    def plot_architecture_layers_polished(self, table: pl.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(9.2, 7.4))
        ax.axis("off")
        colors = ["#d9ead3", "#cfe2f3", "#eadcf8", "#fce5cd", "#d9d2e9"]
        for i, row in enumerate(table.sort("layer_order").to_dicts()):
            y = 0.82 - i * 0.165
            ax.add_patch(Rectangle((0.13, y), 0.74, 0.105, facecolor=colors[i], edgecolor="#333333", linewidth=1.0))
            ax.text(0.16, y + 0.073, f"{row['layer_order']}. {row['layer_name']}", fontsize=12, fontweight="bold", va="center")
            ax.text(0.16, y + 0.044, row["component_line"], fontsize=9.3, va="center")
            ax.text(0.16, y + 0.018, row["metric_line"], fontsize=8.7, va="center", color="#333333")
            if i < table.height - 1:
                ax.annotate("", xy=(0.50, y - 0.045), xytext=(0.50, y - 0.006), arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#333333"})
        ax.set_title("ABM v4 architecture: historical diagnostic stack", fontsize=15, pad=10)
        return fig

    def plot_emissions_decomposition_logic_polished(self, table: pl.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10.2, 5.4))
        ax.axis("off")
        ax.text(0.5, 0.86, r"$E = X \times EI$", ha="center", fontsize=25)
        ax.text(0.5, 0.71, r"$\Delta E = EI\Delta X + X\Delta EI + \Delta X\Delta EI$", ha="center", fontsize=18)
        boxes = [
            (0.08, "Scale effect", r"$EI\Delta X$", "Output change at current intensity"),
            (0.38, "Intensity effect", r"$X\Delta EI$", "Cleaner or dirtier production"),
            (0.68, "Interaction", r"$\Delta X\Delta EI$", "Scale and intensity move together"),
        ]
        for x, title, formula, meaning in boxes:
            ax.add_patch(Rectangle((x, 0.38), 0.24, 0.20, facecolor="#f7f7f7", edgecolor="#333333"))
            ax.text(x + 0.12, 0.525, title, ha="center", fontsize=11, fontweight="bold")
            ax.text(x + 0.12, 0.475, formula, ha="center", fontsize=14)
            ax.text(x + 0.12, 0.425, self._wrap(meaning, 25), ha="center", fontsize=8.8)
        ax.add_patch(Rectangle((0.20, 0.15), 0.60, 0.10, facecolor="#fff2cc", edgecolor="#333333"))
        ax.text(0.5, 0.20, "Falling emissions is not automatically green transition.", ha="center", va="center", fontsize=12, fontweight="bold")
        ax.set_title("ABM v4 emissions decomposition logic", fontsize=15, pad=10)
        return fig

    def plot_two_rule_scorecard(self, table: pl.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(11.2, 5.2))
        ax.axis("off")
        headers = ["Validation dimension", "frontier_gap_readiness", "historical_frontier_gap_only"]
        xs = [0.04, 0.36, 0.68]
        widths = [0.30, 0.30, 0.28]
        for x, width, header in zip(xs, widths, headers):
            ax.add_patch(Rectangle((x, 0.86), width, 0.07, facecolor="#eeeeee", edgecolor="#333333"))
            ax.text(x + width / 2, 0.895, self._wrap(header, 24), ha="center", va="center", fontsize=9.5, fontweight="bold")
        colors = {"strong": "#d9ead3", "moderate": "#fff2cc", "weak": "#fce5cd", "blocked": "#f4cccc"}
        for i, row in enumerate(table.to_dicts()):
            y = 0.76 - i * 0.135
            ax.add_patch(Rectangle((xs[0], y), widths[0], 0.105, facecolor="#ffffff", edgecolor="#dddddd"))
            ax.text(xs[0] + 0.015, y + 0.052, row["validation_dimension"], va="center", fontsize=9.2, fontweight="bold")
            for col_index, prefix in enumerate(("frontier_gap_readiness", "historical_frontier_gap_only"), start=1):
                assessment = str(row[f"{prefix}_assessment"])
                metric = str(row[f"{prefix}_metric"] or "metric unavailable")
                ax.add_patch(Rectangle((xs[col_index], y), widths[col_index], 0.105, facecolor=colors.get(assessment, "#eeeeee"), edgecolor="#dddddd"))
                ax.text(xs[col_index] + 0.012, y + 0.067, assessment, va="center", fontsize=9.2, fontweight="bold")
                ax.text(xs[col_index] + 0.012, y + 0.034, self._wrap(metric, 32), va="center", fontsize=7.8, color="#333333")
        ax.set_title("Why ABM v4 retains two historical rules", fontsize=15, pad=10)
        return fig

    def plot_capability_source_coverage_polished(self, table: pl.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(9.5, 3.9))
        labels = table["capability_type"].to_list()
        segments = [("atlas_observed", "#74a9cf"), ("io_imputed", "#a1d99b"), ("unavailable", "#fdae6b")]
        y_positions = list(range(len(labels)))
        left = [0.0] * len(labels)
        totals = table["total"].to_list()
        for column, color in segments:
            values = table[column].to_list()
            ax.barh(y_positions, values, left=left, label=column.replace("_", " "), color=color)
            for y, value, lft, total in zip(y_positions, values, left, totals):
                share = value / total if total else 0.0
                if share >= 0.08:
                    ax.text(lft + value / 2, y, f"{share:.0%}", ha="center", va="center", fontsize=8)
                else:
                    ax.text(lft + value + max(totals) * 0.01, y, f"{share:.0%}", va="center", fontsize=8)
            left = [l + v for l, v in zip(left, values)]
        ax.set_yticks(y_positions, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Country-sector agents")
        ax.set_title("Source-aware capability assignment reduced crude missing-data treatment")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False)
        fig.tight_layout()
        return fig

    def plot_q_energy_mix_quality_boundary_polished(self, table: pl.DataFrame, title: str) -> plt.Figure:
        return self.plot_table_grid(table, title, ["check", "status", "quantitative_evidence", "implication"], [0.24, 0.14, 0.32, 0.30])

    def plot_china_electricity_boundary_case_polished(self, table: pl.DataFrame) -> plt.Figure:
        valid = table.drop_nulls(["year", "fossil_share"])
        fig, axes = plt.subplots(2, 1, figsize=(9.4, 6.1), sharex=True)
        if valid.height >= 2:
            axes[0].plot(valid["year"], valid["fossil_share"], label="fossil share", color="#b45f06", linewidth=2.0)
            axes[0].plot(valid["year"], valid["coal_share"], label="coal share", color="#444444", linewidth=1.8)
            axes[0].plot(valid["year"], valid["clean_electricity_share"], label="clean electricity share", color="#2c7fb8", linewidth=1.8)
            axes[0].set_ylabel("Energy share")
            axes[0].legend(frameon=False, ncol=3, fontsize=8)
            label = str(valid["observed_series_label"][0])
            axes[1].plot(valid["year"], valid["observed_EI_or_rEI"], label=label, color="#238b45", linewidth=2.0)
            if valid["model_error_frontier_gap_readiness"].drop_nulls().len() > 0:
                axes[1].plot(valid["year"], valid["model_error_frontier_gap_readiness"], label="frontier_gap_readiness model error", color="#756bb1")
            if valid["model_error_historical_frontier_gap_only"].drop_nulls().len() > 0:
                axes[1].plot(valid["year"], valid["model_error_historical_frontier_gap_only"], label="historical_frontier_gap_only model error", color="#de2d26")
            axes[1].set_ylabel(label.replace("Observed ", ""))
            axes[1].set_xlabel("Year")
            axes[1].legend(frameon=False, fontsize=8)
            axes[1].text(0.01, -0.32, str(valid["boundary_caption"][0]), transform=axes[1].transAxes, fontsize=8.5)
        else:
            for ax in axes:
                ax.axis("off")
            axes[0].text(0.5, 0.5, "China electricity time-series audit unavailable.", ha="center", va="center")
            axes[1].text(0.5, 0.5, "Boundary interpretation retained without causal proof.", ha="center", va="center")
        fig.suptitle("China electricity as a fuel-mix boundary case", fontsize=14)
        fig.tight_layout()
        return fig

    def plot_abm_v4_to_v5_roadmap_polished(self, table: pl.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12.2, 6.2))
        ax.axis("off")
        headers = ["ABM v4 finding", "Boundary identified", "ABM v5 requirement"]
        xs = [0.05, 0.37, 0.69]
        colors = ["#cfe2f3", "#fce5cd", "#d9ead3"]
        for x, header in zip(xs, headers):
            ax.text(x + 0.12, 0.94, header, ha="center", fontweight="bold", fontsize=10)
        for i, row in enumerate(table.to_dicts()):
            y = 0.78 - i * 0.15
            labels = [row["abm_v4_finding"], row["boundary_identified"], row["abm_v5_requirement"]]
            for x, label, color in zip(xs, labels, colors):
                ax.add_patch(Rectangle((x, y), 0.24, 0.105, facecolor=color, edgecolor="#333333"))
                ax.text(x + 0.12, y + 0.052, self._wrap(label, 28), ha="center", va="center", fontsize=8.6)
            ax.annotate("", xy=(0.37, y + 0.052), xytext=(0.29, y + 0.052), arrowprops={"arrowstyle": "->", "lw": 1.0})
            ax.annotate("", xy=(0.69, y + 0.052), xytext=(0.61, y + 0.052), arrowprops={"arrowstyle": "->", "lw": 1.0})
        ax.set_title("ABM v4 validation failures define the ABM v5 roadmap", fontsize=15)
        return fig

    def plot_table_grid(self, table: pl.DataFrame, title: str, columns: list[str], widths: list[float]) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, max(4.2, 0.58 * table.height + 1.4)))
        ax.axis("off")
        left = 0.04
        top = 0.88
        row_h = min(0.10, 0.72 / max(table.height, 1))
        xs = [left]
        for width in widths[:-1]:
            xs.append(xs[-1] + width)
        for x, width, column in zip(xs, widths, columns):
            ax.add_patch(Rectangle((x, top), width, 0.065, facecolor="#eeeeee", edgecolor="#333333"))
            ax.text(x + width / 2, top + 0.033, column.replace("_", " ").title(), ha="center", va="center", fontsize=9.2, fontweight="bold")
        color_map = {
            "pass": "#d9ead3",
            "strong": "#d9ead3",
            "retained ABM v4 rule": "#d9ead3",
            "partial": "#fff2cc",
            "moderate": "#fff2cc",
            "limited": "#fff2cc",
            "blocked": "#f4cccc",
            "fail": "#f4cccc",
            "not_scenario_ready": "#f4cccc",
            "rejected": "#f4cccc",
            "rejected as ABM v4 rule, retained as evidence": "#fce5cd",
            "diagnostic only": "#d9d2e9",
        }
        for i, row in enumerate(table.select(columns).to_dicts()):
            y = top - (i + 1) * row_h
            status = str(row.get("status", row.get("final_status", "")))
            for x, width, column in zip(xs, widths, columns):
                face = color_map.get(status, "#ffffff") if column in {"status", "final_status"} else "#ffffff"
                ax.add_patch(Rectangle((x, y), width, row_h, facecolor=face, edgecolor="#dddddd"))
                ax.text(x + 0.01, y + row_h / 2, self._wrap(str(row[column]).replace("_", " "), max(14, int(width * 92))), va="center", fontsize=7.7)
        ax.set_title(title, fontsize=15, pad=10)
        return fig

    def write_source_tables(self, tables: dict[str, pl.DataFrame]) -> tuple[Path, ...]:
        """Write polished source tables."""
        paths = []
        for name in POLISHED_SOURCE_TABLE_NAMES:
            path = self.paths.final_tables_polished / name
            tables[name].write_csv(path)
            paths.append(path)
        q_web = self.build_q_energy_mix_quality_boundary_web_source(
            tables["q_energy_mix_quality_boundary_polished_source.csv"]
        )
        web_path = self.paths.final_tables_polished / "q_energy_mix_quality_boundary_web_source.csv"
        q_web.write_csv(web_path)
        paths.append(web_path)
        return tuple(paths)

    def write_plots(self, plots: dict[str, plt.Figure]) -> tuple[Path, ...]:
        """Write polished plots as PNG and SVG."""
        paths = []
        for name in POLISHED_PLOT_NAMES:
            figure = plots[name]
            for suffix in (".png", ".svg"):
                path = self.paths.final_plots_polished / f"{name}{suffix}"
                figure.savefig(path, bbox_inches="tight", dpi=180)
                paths.append(path)
            plt.close(figure)
        return tuple(paths)

    def copy_plots_to_output_directory(self, plot_paths: tuple[Path, ...]) -> tuple[Path, ...]:
        """Copy polished plots to the final polished output folder."""
        self.paths.outputs_plots_abm_v4_final_polished.mkdir(parents=True, exist_ok=True)
        copied = []
        for path in plot_paths:
            destination = self.paths.outputs_plots_abm_v4_final_polished / path.name
            copy2(path, destination)
            copied.append(destination)
        return tuple(copied)

    def build_plot_index(
        self,
        source_table_paths: tuple[Path, ...],
        plot_paths: tuple[Path, ...],
        copied_plot_paths: tuple[Path, ...],
    ) -> pl.DataFrame:
        """Build the polished plot artifact index."""
        rows = []
        for path in source_table_paths:
            rows.append(self._index_row("polished_source_table", path, "Source table for a Phase 29C polished plot."))
        for path in plot_paths:
            rows.append(self._index_row("polished_plot", path, "Polished ABM v4 final figure."))
        for path in copied_plot_paths:
            rows.append(self._index_row("polished_plot_copy", path, "Copied polished plot for report/web assets."))
        for path in (*self.required_input_paths(), *self.optional_input_paths()):
            if path.exists():
                rows.append(self._index_row("input_source", path, "Input used or available to Phase 29C."))
        return pl.DataFrame(rows)

    def _scorecard_row(
        self,
        validation_dimension: str,
        fgr_assessment: str,
        fgr_metric: str | None,
        hfg_assessment: str,
        hfg_metric: str | None,
        evidence_source: str,
        interpretation: str,
    ) -> dict[str, str | None]:
        return {
            "validation_dimension": validation_dimension,
            "frontier_gap_readiness_assessment": fgr_assessment,
            "frontier_gap_readiness_metric": fgr_metric,
            "historical_frontier_gap_only_assessment": hfg_assessment,
            "historical_frontier_gap_only_metric": hfg_metric,
            "evidence_source": evidence_source,
            "interpretation": interpretation,
        }

    def _metrics_by_rule(self) -> dict[str, dict[str, str]]:
        metrics: dict[str, dict[str, str]] = {}
        path = self.paths.multiyear_base_model_comparison_csv_path
        if path.exists():
            for row in pl.read_csv(path).to_dicts():
                rule = str(row.get("model_variant", ""))
                metrics[rule] = {
                    "aggregate": self._format_metric(row, ("mean_yearly_aggregate_emissions_pct_error", "latest_aggregate_emissions_pct_error"), "aggregate pct error"),
                    "rEI": self._format_metric(row, ("rEI_MAE", "mean_EI_error"), "rEI MAE"),
                    "electricity": self._format_metric(row, ("electricity_error", "china_electricity_error", "electricity_pct_error"), "electricity error"),
                }
        return metrics

    def _format_metric(self, row: dict[str, object], names: tuple[str, ...], label: str) -> str | None:
        for name in names:
            value = row.get(name)
            if isinstance(value, (int, float)):
                return f"{label}: {abs(float(value)):.3g}"
        return None

    def _shorten(self, value: object, max_len: int) -> str:
        text = str(value)
        if len(text) <= max_len:
            return text
        return text[: max_len - 1].rstrip() + "."


@dataclass(frozen=True)
class ABMV4NarrativePlotResult:
    """Paths written by the narrative-grade final ABM v4 plot builder."""

    source_table_paths: tuple[Path, ...]
    plot_paths: tuple[Path, ...]
    copied_plot_paths: tuple[Path, ...]
    plot_index_path: Path


class ABMV4NarrativePlotBuilder:
    """Build Phase 29B narrative-grade ABM v4 final figures and source tables."""

    def __init__(self, paths: ABMV4Paths) -> None:
        self.paths = paths

    def run(self, *, write_outputs: bool, copy_output_plots: bool = True) -> ABMV4NarrativePlotResult:
        """Build narrative figures and tables, writing only when explicitly requested."""
        self.validate_required_inputs()
        tables = self.build_source_tables()
        plots = self.build_plots(tables)
        if not write_outputs:
            for figure in plots.values():
                plt.close(figure)
            return ABMV4NarrativePlotResult((), (), (), self.paths.final_narrative_plot_index_path)

        self.paths.ensure_final_narrative_artifact_directories()
        table_paths = self.write_source_tables(tables)
        plot_paths = self.write_plots(plots)
        copied_paths: tuple[Path, ...] = ()
        if copy_output_plots:
            copied_paths = self.copy_plots_to_output_directory(plot_paths)
        index = self.build_plot_index(table_paths, plot_paths, copied_paths)
        index.write_csv(self.paths.final_narrative_plot_index_path)
        return ABMV4NarrativePlotResult(table_paths, plot_paths, copied_paths, self.paths.final_narrative_plot_index_path)

    def validate_required_inputs(self) -> None:
        """Fail clearly if required Phase 28 final validation files are missing."""
        missing = [path for path in self.required_input_paths() if not path.exists()]
        if missing:
            formatted = "\n".join(f"- {self._relative(path)}" for path in missing)
            raise FileNotFoundError(
                "Cannot build ABM v4 narrative plots because required Phase 28 outputs are missing.\n"
                f"{formatted}\n"
                "Run: python scripts/run_abm_v4_base.py --finalize-abm-v4 --create-output-dirs"
            )

    def required_input_paths(self) -> tuple[Path, ...]:
        """Return required inputs for Phase 29B."""
        return (
            self.paths.final_surviving_rule_comparison_path,
            self.paths.final_validation_objective_matrix_path,
            self.paths.final_rejected_mechanism_register_path,
            self.paths.final_scenario_readiness_assessment_path,
            self.paths.final_abm_v5_research_agenda_path,
            self.paths.final_abm_v4_hypothesis_status_path,
            self.paths.final_model_boundary_statement_path,
        )

    def optional_input_paths(self) -> tuple[Path, ...]:
        """Return optional evidence inputs used opportunistically by Phase 29B."""
        return (
            self.paths.final_tables / "final_two_rule_summary.csv",
            self.paths.final_tables / "final_mechanism_status.csv",
            self.paths.final_tables / "final_scenario_blockers.csv",
            self.paths.final_tables / "final_abm_v5_priorities.csv",
            self.paths.multiyear_base_model_comparison_csv_path,
            self.paths.multiyear_error_summary_path,
            self.paths.multiyear_EID_diagnostic_comparison_path,
            self.paths.adaptive_EID_model_comparison_path,
            self.paths.io_capability_robustness_path,
            self.paths.io_capability_threshold_sensitivity_path,
            self.paths.emissions_transition_variant_results_path,
            self.paths.q_energy_mix_quality_audit_path,
            self.paths.q_energy_mix_recommendation_path,
            self.paths.q_energy_mix_hypothesis_tests_path,
            self.paths.q_energy_mix_china_electricity_audit_path,
            self.paths.q_energy_mix_predictor_screening_path,
            self.paths.transition_rule_aggregate_contribution_path,
            self.paths.electricity_transition_regime_recommendation_path,
        )

    def build_source_tables(self) -> dict[str, pl.DataFrame]:
        """Build every narrative plot source table in memory."""
        return {
            "architecture_layers_source.csv": self.build_architecture_layers_source(),
            "emissions_decomposition_logic_source.csv": self.build_emissions_decomposition_logic_source(),
            "two_rule_metric_tradeoff_source.csv": self.build_two_rule_metric_tradeoff_source(),
            "mechanism_decision_tree_source.csv": self.build_mechanism_decision_tree_source(),
            "capability_source_coverage_source.csv": self.build_capability_source_coverage_source(),
            "q_energy_mix_quality_boundary_source.csv": self.build_q_energy_mix_quality_boundary_source(),
            "china_electricity_boundary_case_source.csv": self.build_china_electricity_boundary_case_source(),
            "scenario_readiness_checklist_source.csv": self.build_scenario_readiness_checklist_source(),
            "abm_v4_to_v5_roadmap_source.csv": self.build_abm_v4_to_v5_roadmap_source(),
            "hypothesis_status_table_source.csv": self.build_hypothesis_status_table_source(),
        }

    def build_architecture_layers_source(self) -> pl.DataFrame:
        """Build the architecture source table."""
        return pl.DataFrame(
            [
                {
                    "layer_order": 1,
                    "layer_name": "Historical data inputs",
                    "components": "Eora26 output and transactions; emissions intensity; Atlas and IO capabilities; Q energy rows as diagnostic only",
                    "key_metric": "1995-2016 historical sources",
                    "interpretation": "ABM v4 starts from observed historical production-network and emissions data.",
                },
                {
                    "layer_order": 2,
                    "layer_name": "Country-sector state panel",
                    "components": "Country-sector nodes with output, EI, capabilities, and ecosystem fields",
                    "key_metric": "4,915 agents; 108,130 country-sector-year rows; 1995-2016",
                    "interpretation": "The model is defined at country-sector-agent resolution.",
                },
                {
                    "layer_order": 3,
                    "layer_name": "Production network and ecosystems",
                    "components": "supplier-buyer edges; opportunity sets; productive ecosystem mapping",
                    "key_metric": "Observed supplier-buyer production-network foundation",
                    "interpretation": "Transition mechanisms are evaluated in a networked production context.",
                },
                {
                    "layer_order": 4,
                    "layer_name": "Behavioural diagnostic layers",
                    "components": "supplier adaptation; capability exposure; emissions-transition rules",
                    "key_metric": "Mechanisms tested, retained, or rejected",
                    "interpretation": "ABM v4 tests mechanisms as historical diagnostics, not scenario policies.",
                },
                {
                    "layer_order": 5,
                    "layer_name": "Historical validation and model boundary",
                    "components": "two surviving rules; rejected mechanisms; not scenario-ready",
                    "key_metric": "Completed historical diagnostic framework",
                    "interpretation": "Validation identifies the ABM v4 boundary and motivates ABM v5.",
                },
            ]
        )

    def build_emissions_decomposition_logic_source(self) -> pl.DataFrame:
        """Build the emissions decomposition logic source table."""
        return pl.DataFrame(
            [
                {
                    "component": "Emissions identity",
                    "formula": "E = X * EI",
                    "interpretation": "Emissions combine production scale and emissions intensity.",
                    "transition_meaning": "Emissions changes are not interpretable without separating X and EI.",
                },
                {
                    "component": "Production-scale effect",
                    "formula": "EI * Delta X",
                    "interpretation": "Emissions change caused by output change at existing intensity.",
                    "transition_meaning": "Declining output can reduce emissions without structural greening.",
                },
                {
                    "component": "Emissions-intensity effect",
                    "formula": "X * Delta EI",
                    "interpretation": "Emissions change caused by cleaner or dirtier production intensity.",
                    "transition_meaning": "This is the core signal for genuine green transition.",
                },
                {
                    "component": "Interaction effect",
                    "formula": "Delta X * Delta EI",
                    "interpretation": "Joint scale and intensity movement.",
                    "transition_meaning": "Growth with cleaner intensity and decline with dirtier intensity require different readings.",
                },
                {
                    "component": "ABM v4 interpretation rule",
                    "formula": "Delta E = EI Delta X + X Delta EI + Delta X Delta EI",
                    "interpretation": "Falling emissions is not automatically green transition.",
                    "transition_meaning": "Genuine transition requires EI improvement without relying on output collapse.",
                },
            ]
        )

    def build_two_rule_metric_tradeoff_source(self) -> pl.DataFrame:
        """Build quantitative two-rule tradeoff data, falling back to diagnostic scores."""
        comparison_path = self.paths.multiyear_base_model_comparison_csv_path
        if comparison_path.exists():
            comparison = pl.read_csv(comparison_path)
            return self._two_rule_metrics_from_comparison(comparison, comparison_path.name)

        rows = []
        objective_matrix = pl.read_csv(self.paths.final_validation_objective_matrix_path)
        variants = {
            "frontier_gap_readiness": "frontier_gap_readiness_assessment",
            "historical_frontier_gap_only": "historical_frontier_gap_only_assessment",
            "fixed_EID_diagnostic": "EID_assessment",
        }
        for variant, column in variants.items():
            if column not in objective_matrix.columns:
                continue
            scores = [self._diagnostic_error_score(value) for value in objective_matrix[column].to_list()]
            mean_score = sum(scores) / max(len(scores), 1)
            retained = "retained" if variant in {"frontier_gap_readiness", "historical_frontier_gap_only"} else "rejected_as_rule"
            rows.append(
                {
                    "model_variant": variant,
                    "transition_error_metric": mean_score,
                    "aggregate_error_metric": mean_score,
                    "electricity_error_metric_if_available": None,
                    "source_file": "final_validation_objective_matrix.csv",
                    "retained_status": retained,
                    "interpretation": "diagnostic score, not direct error metric",
                }
            )
        return pl.DataFrame(rows)

    def _two_rule_metrics_from_comparison(self, comparison: pl.DataFrame, source_file: str) -> pl.DataFrame:
        rows = []
        for row in comparison.to_dicts():
            variant = str(row.get("model_variant", ""))
            transition_metric = self._first_number(row, ("emissions_weighted_rEI_MAE", "rEI_MAE", "mean_rEI_abs_error", "mean_EI_error"))
            aggregate_metric = self._first_number(row, ("mean_yearly_aggregate_emissions_pct_error", "latest_aggregate_emissions_pct_error", "aggregate_emissions_pct_error"))
            electricity_metric = self._first_number(row, ("electricity_error", "china_electricity_error", "electricity_pct_error"))
            retained = "retained" if variant in {"frontier_gap_readiness", "historical_frontier_gap_only"} else "rejected_as_rule"
            if "EID" in variant or "eid" in variant:
                retained = "rejected_as_rule"
            rows.append(
                {
                    "model_variant": variant,
                    "transition_error_metric": transition_metric,
                    "aggregate_error_metric": aggregate_metric,
                    "electricity_error_metric_if_available": electricity_metric,
                    "source_file": source_file,
                    "retained_status": retained,
                    "interpretation": self._tradeoff_interpretation(variant),
                }
            )
        return pl.DataFrame(rows)

    def build_mechanism_decision_tree_source(self) -> pl.DataFrame:
        """Build the mechanism decision tree source table."""
        rejected = pl.read_csv(self.paths.final_rejected_mechanism_register_path)
        return pl.DataFrame(
            [
                self._mechanism_row("raw log EI rule", "legacy rule", "rejected", "theoretically fragile / sign issue", "baseline foil", "none", "not_scenario_ready", rejected, "legacy_raw_log emissions rule"),
                self._mechanism_row("frontier_gap_readiness", "surviving rule", "retained", "best aggregate emissions plausibility", "aggregate-safe baseline", "ABM v4 historical diagnostic baseline", "not_scenario_ready"),
                self._mechanism_row("historical_frontier_gap_only", "surviving rule", "retained", "cleaner frontier-gap transition interpretation", "transition-mechanism benchmark", "ABM v4 mechanism benchmark", "not_scenario_ready"),
                self._mechanism_row("fixed EID dampener", "EID branch", "rejected as ABM v4 rule", "failed as transition rule", "ontology evidence", "ABM v5 structural agent types", "not_scenario_ready", rejected, "fixed EID dampener"),
                self._mechanism_row("adaptive EID dampener", "EID branch", "rejected", "forward validation / overfitting risk", "calibration warning and ontology evidence", "ABM v5 ontology design input", "not_scenario_ready", rejected, "adaptive EID dampener"),
                self._mechanism_row("Q energy mix country-sector rule", "energy branch", "rejected as ABM v4 rule", "node-level Q quality limits", "aggregate diagnostic and ABM v5 evidence", "cleaner energy/fuel data", "not_scenario_ready", rejected, "Q energy mix country-sector transition rule"),
                self._mechanism_row("historical residual", "residual branch", "diagnostic only", "not scenario-facing", "diagnostic benchmark", "feature discovery only", "not_scenario_ready", rejected, "historical residual as scenario-facing rule"),
            ]
        )

    def build_capability_source_coverage_source(self) -> pl.DataFrame:
        """Build source-aware capability coverage table."""
        rows = [
            {
                "capability_type": "General capability",
                "atlas_observed": 2924,
                "io_imputed": 1548,
                "unavailable": 443,
                "source_file": "Phase 29B acceptance constants",
                "interpretation": "IO imputation substantially extends general capability coverage.",
            },
            {
                "capability_type": "Green capability",
                "atlas_observed": 2924,
                "io_imputed": 1316,
                "unavailable": 675,
                "source_file": "Phase 29B acceptance constants",
                "interpretation": "Green capability still has larger unavailable coverage after IO imputation.",
            },
        ]
        source_path = self.paths.io_capability_robustness_path
        if source_path.exists():
            table = pl.read_csv(source_path)
            rows = self._capability_rows_from_optional_table(table, source_path.name) or rows
        for row in rows:
            total = row["atlas_observed"] + row["io_imputed"] + row["unavailable"]
            row["total"] = total
            row["atlas_share"] = row["atlas_observed"] / total
            row["io_imputed_share"] = row["io_imputed"] / total
            row["unavailable_share"] = row["unavailable"] / total
        return pl.DataFrame(rows)

    def build_q_energy_mix_quality_boundary_source(self) -> pl.DataFrame:
        """Build Q energy mix quality boundary checklist table."""
        quality = self._read_optional_csv(self.paths.q_energy_mix_quality_audit_path)
        recommendation = self._read_optional_csv(self.paths.q_energy_mix_recommendation_path)
        screening = self._read_optional_csv(self.paths.q_energy_mix_predictor_screening_path)
        rows = [
            self._q_check("all 9 Q energy rows found", "pass", "9 mapped energy rows expected", "Q rows can be audited as energy diagnostics.", "q_energy_source_inventory.csv"),
            self._q_check("broad country-sector coverage", "pass", self._q_panel_rows_note(quality), "Coverage is broad enough for aggregate diagnostics.", "q_energy_mix_quality_audit.csv"),
            self._q_check("aggregate diagnostic value", "pass", self._recommendation_note(recommendation), "Aggregate energy-mix evidence is retained.", "q_energy_mix_recommendation.csv"),
            self._q_check("China electricity fuel-mix signal", "partial", "China electricity fossil/coal shares are observable.", "Boundary evidence, not a causal claim.", "q_energy_mix_china_electricity_audit.csv"),
            self._q_check("valid node-level shares", "fail", self._invalid_share_note(quality), "Reject node-level country-sector rule integration.", "q_energy_mix_quality_audit.csv"),
            self._q_check("no negative values", "fail", self._negative_count_note(quality), "Negative source values prevent clean behavioural rule use.", "q_energy_mix_quality_audit.csv"),
            self._q_check("no severe aggregate flags", "partial", "Aggregate plausibility remains diagnostic rather than rule-grade.", "Use only as aggregate evidence.", "q_energy_mix_aggregate_plausibility.csv"),
            self._q_check("strong predictive power", "partial", self._best_predictor_note(screening), "Predictor signal motivates ABM v5 energy/fuel structure.", "q_energy_mix_predictor_screening.csv"),
        ]
        return pl.DataFrame(rows)

    def build_china_electricity_boundary_case_source(self) -> pl.DataFrame:
        """Build China electricity boundary-case time-series or fallback table."""
        path = self.paths.q_energy_mix_china_electricity_audit_path
        columns = [
            "year",
            "coal_share",
            "fossil_share",
            "clean_electricity_share",
            "observed_EI_or_rEI",
            "model_error_frontier_gap_readiness",
            "model_error_historical_frontier_gap_only",
            "interpretation",
        ]
        if path.exists():
            source = pl.read_csv(path)
            rows = []
            for row in source.to_dicts():
                rows.append(
                    {
                        "year": row.get("year"),
                        "coal_share": row.get("coal_share"),
                        "fossil_share": row.get("fossil_share"),
                        "clean_electricity_share": row.get("clean_electricity_share"),
                        "observed_EI_or_rEI": row.get("observed_rEI", row.get("EI_observed")),
                        "model_error_frontier_gap_readiness": row.get("model_error_frontier_gap_readiness"),
                        "model_error_historical_frontier_gap_only": row.get("model_error_historical_frontier_gap_only"),
                        "interpretation": "Fuel-mix boundary case; do not claim causality.",
                    }
                )
            return pl.DataFrame(rows, schema=columns)
        return pl.DataFrame(
            [
                {
                    "year": None,
                    "coal_share": None,
                    "fossil_share": None,
                    "clean_electricity_share": None,
                    "observed_EI_or_rEI": None,
                    "model_error_frontier_gap_readiness": None,
                    "model_error_historical_frontier_gap_only": None,
                    "interpretation": "Optional China electricity audit missing; plot uses fallback boundary statement.",
                }
            ],
            schema=columns,
        )

    def build_scenario_readiness_checklist_source(self) -> pl.DataFrame:
        """Build scenario-readiness checklist source table."""
        readiness = pl.read_csv(self.paths.final_scenario_readiness_assessment_path)
        wanted = [
            "emissions_transition_rule",
            "production_dynamics",
            "supplier_substitution",
            "capability_dynamics",
            "electricity_energy_system",
            "policy_institutional_variables",
            "data_quality",
            "validation_metrics",
            "interpretation_risk",
            "overall_scenario_readiness",
        ]
        rows = []
        for dimension in wanted:
            matching = readiness.filter(pl.col("readiness_dimension") == dimension)
            if matching.is_empty():
                rows.append(
                    {
                        "readiness_dimension": dimension,
                        "status": "blocked" if dimension != "overall_scenario_readiness" else "not_scenario_ready",
                        "blocking_issue": "Not cleared by Phase 28 final assessment.",
                        "required_future_work": "ABM v5 mechanism and validation work.",
                        "interpretation": "Missing explicit readiness row treated as unresolved, not as ready.",
                    }
                )
            else:
                row = matching.row(0, named=True)
                rows.append(
                    {
                        "readiness_dimension": dimension,
                        "status": row.get("status", ""),
                        "blocking_issue": row.get("blocking_issue", row.get("evidence", "")),
                        "required_future_work": row.get("required_future_work", ""),
                        "interpretation": row.get("interpretation", row.get("evidence", "")),
                    }
                )
        return pl.DataFrame(rows)

    def build_abm_v4_to_v5_roadmap_source(self) -> pl.DataFrame:
        """Build the ABM v4 to ABM v5 roadmap source table."""
        return pl.DataFrame(
            [
                {
                    "abm_v4_finding": "Q energy mix aggregate-only",
                    "boundary_identified": "Node-level Q shares are not rule-grade.",
                    "abm_v5_requirement": "cleaner energy/fuel data",
                    "candidate_agent_type": "energy infrastructure agents",
                    "required_data": "Fuel use, generation, capacity, and country-sector energy balances.",
                    "validation_test": "Fuel-mix rules improve electricity and high-emissions validation out of sample.",
                },
                {
                    "abm_v4_finding": "electricity boundary",
                    "boundary_identified": "Electricity errors expose missing fuel/policy mechanisms.",
                    "abm_v5_requirement": "policy/investment regime",
                    "candidate_agent_type": "policy/institutional layer",
                    "required_data": "Policy, investment, subsidy, regulation, and price regime data.",
                    "validation_test": "Policy regimes explain transition accelerations without residual forcing.",
                },
                {
                    "abm_v4_finding": "EID ontology evidence",
                    "boundary_identified": "EID classifies structural node roles but fails as a dampener.",
                    "abm_v5_requirement": "structural agent types",
                    "candidate_agent_type": "explicit agent ontology",
                    "required_data": "Dependence, essential-input, infrastructure, and sector role metrics.",
                    "validation_test": "Agent types improve heterogeneous transition dynamics.",
                },
                {
                    "abm_v4_finding": "historical production forcing",
                    "boundary_identified": "Counterfactual production does not emerge endogenously.",
                    "abm_v5_requirement": "endogenous production dynamics",
                    "candidate_agent_type": "production and demand agents",
                    "required_data": "Demand, capacity, substitution, input requirements, and propagation data.",
                    "validation_test": "Production paths validate without historical output forcing.",
                },
                {
                    "abm_v4_finding": "high-emissions infrastructure",
                    "boundary_identified": "Long-lived assets are not represented.",
                    "abm_v5_requirement": "capital-stock inertia",
                    "candidate_agent_type": "capital-stock and asset-turnover agents",
                    "required_data": "Plant age, installed capacity, investment, retirement, and asset lifetime data.",
                    "validation_test": "Stock turnover constraints explain slow high-emissions transition.",
                },
            ]
        )

    def build_hypothesis_status_table_source(self) -> pl.DataFrame:
        """Build grouped hypothesis status table source."""
        source = pl.read_csv(self.paths.final_abm_v4_hypothesis_status_path)
        wanted = [
            "frontier_gap_readiness_aggregate_safe",
            "historical_frontier_gap_transition_benchmark",
            "EID_as_transition_dampener",
            "EID_as_ontology_signal",
            "adaptive_EID_as_ABM_v4_rule",
            "Q_energy_mix_as_country_sector_rule",
            "Q_energy_mix_as_aggregate_diagnostic",
            "China_electricity_missing_fuel_mix_mechanism",
            "ABM_v4_scenario_ready",
            "ABM_v5_needs_energy_policy_agent_ontology",
        ]
        fallback = {
            "adaptive_EID_as_ABM_v4_rule": ("not_supported", "Adaptive EID rejected for ABM v4 rule use.", "Retain only as calibration warning."),
            "China_electricity_missing_fuel_mix_mechanism": ("diagnostic_only", "China electricity remains a boundary case.", "Motivates energy/fuel structure."),
            "ABM_v5_needs_energy_policy_agent_ontology": ("supported", "ABM v5 agenda follows from final validation failures.", "Implement ABM v5 mechanisms later."),
        }
        rows = []
        for hypothesis in wanted:
            matching = source.filter(pl.col("hypothesis") == hypothesis)
            if matching.is_empty():
                status, evidence, implication = fallback.get(hypothesis, ("diagnostic_only", "No explicit Phase 28 row.", "Treat as unresolved."))
            else:
                row = matching.row(0, named=True)
                status = row.get("status", "")
                evidence = row.get("short_evidence", row.get("evidence", ""))
                implication = row.get("implication", row.get("interpretation", ""))
            rows.append(
                {
                    "hypothesis": hypothesis,
                    "status": status,
                    "short_evidence": evidence,
                    "implication": implication,
                }
            )
        return pl.DataFrame(rows)

    def build_plots(self, tables: dict[str, pl.DataFrame]) -> dict[str, plt.Figure]:
        """Build every narrative plot from source tables."""
        return {
            "abm_v4_architecture_layers": self.plot_architecture_layers(tables["architecture_layers_source.csv"]),
            "abm_v4_emissions_decomposition_logic": self.plot_emissions_decomposition_logic(tables["emissions_decomposition_logic_source.csv"]),
            "abm_v4_two_rule_metric_tradeoff": self.plot_two_rule_metric_tradeoff(tables["two_rule_metric_tradeoff_source.csv"]),
            "abm_v4_mechanism_decision_tree": self.plot_mechanism_decision_tree(tables["mechanism_decision_tree_source.csv"]),
            "abm_v4_capability_source_coverage": self.plot_capability_source_coverage(tables["capability_source_coverage_source.csv"]),
            "abm_v4_q_energy_mix_quality_boundary": self.plot_q_energy_mix_quality_boundary(tables["q_energy_mix_quality_boundary_source.csv"]),
            "abm_v4_china_electricity_boundary_case": self.plot_china_electricity_boundary_case(tables["china_electricity_boundary_case_source.csv"]),
            "abm_v4_scenario_readiness_checklist": self.plot_scenario_readiness_checklist(tables["scenario_readiness_checklist_source.csv"]),
            "abm_v4_to_v5_roadmap": self.plot_abm_v4_to_v5_roadmap(tables["abm_v4_to_v5_roadmap_source.csv"]),
            "abm_v4_hypothesis_status_table": self.plot_hypothesis_status_table(tables["hypothesis_status_table_source.csv"]),
        }

    def plot_architecture_layers(self, table: pl.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 7.2))
        ax.axis("off")
        colors = ["#d9ead3", "#cfe2f3", "#eadcf8", "#fce5cd", "#d9d2e9"]
        for i, row in enumerate(table.sort("layer_order").to_dicts()):
            y = 0.83 - i * 0.17
            rect = Rectangle((0.08, y), 0.84, 0.115, facecolor=colors[i], edgecolor="#303030", linewidth=1.1)
            ax.add_patch(rect)
            ax.text(0.11, y + 0.079, f"{row['layer_order']}. {row['layer_name']}", fontsize=12, fontweight="bold", va="center")
            ax.text(0.11, y + 0.047, self._wrap(row["components"], 95), fontsize=9, va="center")
            ax.text(0.11, y + 0.018, row["key_metric"], fontsize=8.5, va="center", color="#333333")
            if i < table.height - 1:
                ax.annotate("", xy=(0.5, y - 0.018), xytext=(0.5, y - 0.055), arrowprops={"arrowstyle": "->", "lw": 1.2})
        ax.set_title("ABM v4 architecture: historical diagnostic stack", fontsize=15, pad=12)
        return fig

    def plot_emissions_decomposition_logic(self, table: pl.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 5.6))
        ax.axis("off")
        ax.text(0.5, 0.86, r"$E = X \times EI$", ha="center", fontsize=24)
        ax.text(0.5, 0.70, r"$\Delta E = EI\,\Delta X + X\,\Delta EI + \Delta X\,\Delta EI$", ha="center", fontsize=18)
        boxes = [
            (0.10, 0.42, "production-scale effect", "EI * Delta X", "Output decline can lower emissions without greening."),
            (0.39, 0.42, "emissions-intensity effect", "X * Delta EI", "The core evidence for cleaner production."),
            (0.68, 0.42, "interaction effect", "Delta X * Delta EI", "Scale and intensity can move together."),
        ]
        for x, y, title, formula, meaning in boxes:
            ax.add_patch(Rectangle((x, y), 0.23, 0.18, facecolor="#f4f4f4", edgecolor="#303030"))
            ax.text(x + 0.115, y + 0.125, title, ha="center", fontsize=10, fontweight="bold")
            ax.text(x + 0.115, y + 0.083, formula, ha="center", fontsize=10)
            ax.text(x + 0.115, y + 0.035, self._wrap(meaning, 28), ha="center", fontsize=8)
        ax.text(0.5, 0.20, "Falling emissions is not automatically green transition.", ha="center", fontsize=12, fontweight="bold")
        ax.text(0.5, 0.13, "Genuine transition requires emissions-intensity improvement without relying on output collapse.", ha="center", fontsize=10)
        ax.set_title("ABM v4 emissions decomposition logic", fontsize=15, pad=10)
        return fig

    def plot_two_rule_metric_tradeoff(self, table: pl.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8.5, 6))
        rows = table.drop_nulls(["transition_error_metric", "aggregate_error_metric"]).to_dicts()
        for row in rows:
            color = "#2c7fb8" if row["retained_status"] == "retained" else "#bdbdbd"
            marker = "o" if row["retained_status"] == "retained" else "x"
            ax.scatter(row["transition_error_metric"], row["aggregate_error_metric"], s=90, color=color, marker=marker)
            ax.text(row["transition_error_metric"], row["aggregate_error_metric"], f"  {row['model_variant']}", va="center", fontsize=8)
        ax.set_xlabel("Transition-mechanism error")
        ax.set_ylabel("Aggregate emissions error")
        ax.set_title("Two surviving rules optimize different validation objectives")
        ax.text(0.02, 0.96, "lower-left is better", transform=ax.transAxes, fontsize=9, va="top")
        ax.grid(True, alpha=0.25)
        if rows and "diagnostic score" in str(rows[0].get("interpretation", "")):
            ax.text(0.02, 0.06, "Diagnostic score, not direct error metric.", transform=ax.transAxes, fontsize=9)
        fig.tight_layout()
        return fig

    def plot_mechanism_decision_tree(self, table: pl.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(11, 6.2))
        ax.axis("off")
        status_colors = {
            "retained": "#d9ead3",
            "rejected as ABM v4 rule": "#fce5cd",
            "rejected": "#f4cccc",
            "diagnostic only": "#d9d2e9",
        }
        for i, row in enumerate(table.to_dicts()):
            y = 0.86 - i * 0.115
            color = status_colors.get(row["status"], "#eeeeee")
            ax.add_patch(Rectangle((0.04, y), 0.25, 0.075, facecolor=color, edgecolor="#303030"))
            ax.add_patch(Rectangle((0.32, y), 0.20, 0.075, facecolor=color, edgecolor="#303030"))
            ax.add_patch(Rectangle((0.55, y), 0.39, 0.075, facecolor="#ffffff", edgecolor="#303030"))
            ax.text(0.055, y + 0.038, self._wrap(row["mechanism"], 28), va="center", fontsize=8.5, fontweight="bold")
            ax.text(0.42, y + 0.038, self._wrap(row["status"], 24), va="center", ha="center", fontsize=8.5)
            ax.text(0.57, y + 0.050, self._wrap(row["reason"], 55), va="center", fontsize=7.8)
            ax.text(0.57, y + 0.020, self._wrap(row["retained_value"], 55), va="center", fontsize=7.8, color="#333333")
        ax.set_title("ABM v4 mechanism testing decision tree", fontsize=15)
        return fig

    def plot_capability_source_coverage(self, table: pl.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(9, 3.8))
        labels = table["capability_type"].to_list()
        segments = [("atlas_observed", "#74a9cf"), ("io_imputed", "#a1d99b"), ("unavailable", "#fdae6b")]
        y_positions = range(len(labels))
        left = [0.0] * len(labels)
        for column, color in segments:
            values = table[column].to_list()
            ax.barh(list(y_positions), values, left=left, label=column.replace("_", " "), color=color)
            left = [l + v for l, v in zip(left, values)]
        ax.set_yticks(list(y_positions), labels)
        ax.invert_yaxis()
        ax.set_xlabel("Country-sector agents")
        ax.set_title("Capability source coverage after IO imputation")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False)
        fig.tight_layout()
        return fig

    def plot_q_energy_mix_quality_boundary(self, table: pl.DataFrame) -> plt.Figure:
        return self._plot_checklist_matrix(table, "check", "status", "implication", "Q energy mix quality boundary")

    def plot_china_electricity_boundary_case(self, table: pl.DataFrame) -> plt.Figure:
        valid = table.drop_nulls(["year", "fossil_share"])
        fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
        if valid.height >= 2:
            axes[0].plot(valid["year"], valid["fossil_share"], label="fossil share", color="#b45f06", linewidth=2)
            if "coal_share" in valid.columns:
                axes[0].plot(valid["year"], valid["coal_share"], label="coal share", color="#444444", linewidth=1.8)
            axes[0].plot(valid["year"], valid["clean_electricity_share"], label="clean electricity share", color="#2c7fb8", linewidth=1.8)
            axes[0].set_ylabel("Share")
            axes[0].legend(frameon=False, ncol=3, fontsize=8)
            axes[1].plot(valid["year"], valid["observed_EI_or_rEI"], label="observed EI or rEI", color="#238b45", linewidth=2)
            if valid["model_error_frontier_gap_readiness"].drop_nulls().len() > 0:
                axes[1].plot(valid["year"], valid["model_error_frontier_gap_readiness"], label="frontier_gap_readiness error", color="#756bb1")
            if valid["model_error_historical_frontier_gap_only"].drop_nulls().len() > 0:
                axes[1].plot(valid["year"], valid["model_error_historical_frontier_gap_only"], label="historical_frontier_gap_only error", color="#de2d26")
            axes[1].set_ylabel("Observed / error")
            axes[1].set_xlabel("Year")
            axes[1].legend(frameon=False, fontsize=8)
        else:
            for ax in axes:
                ax.axis("off")
            axes[0].text(0.5, 0.5, "Optional China electricity time-series audit unavailable.", ha="center", va="center")
            axes[1].text(0.5, 0.5, "Boundary interpretation retained without causal claim.", ha="center", va="center")
        fig.suptitle("China electricity as a fuel-mix boundary case", fontsize=14)
        fig.tight_layout()
        return fig

    def plot_scenario_readiness_checklist(self, table: pl.DataFrame) -> plt.Figure:
        return self._plot_checklist_matrix(table, "readiness_dimension", "status", "blocking_issue", "Scenario readiness checklist")

    def plot_abm_v4_to_v5_roadmap(self, table: pl.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 6.5))
        ax.axis("off")
        for i, row in enumerate(table.to_dicts()):
            y = 0.83 - i * 0.17
            xs = [0.04, 0.36, 0.68]
            labels = [row["abm_v4_finding"], row["boundary_identified"], row["abm_v5_requirement"]]
            colors = ["#cfe2f3", "#fce5cd", "#d9ead3"]
            for x, label, color in zip(xs, labels, colors):
                ax.add_patch(Rectangle((x, y), 0.25, 0.105, facecolor=color, edgecolor="#303030"))
                ax.text(x + 0.125, y + 0.052, self._wrap(label, 30), ha="center", va="center", fontsize=8.5)
            ax.annotate("", xy=(0.36, y + 0.052), xytext=(0.29, y + 0.052), arrowprops={"arrowstyle": "->"})
            ax.annotate("", xy=(0.68, y + 0.052), xytext=(0.61, y + 0.052), arrowprops={"arrowstyle": "->"})
        ax.text(0.165, 0.96, "ABM v4 finding", ha="center", fontweight="bold")
        ax.text(0.485, 0.96, "Boundary identified", ha="center", fontweight="bold")
        ax.text(0.805, 0.96, "ABM v5 requirement", ha="center", fontweight="bold")
        ax.set_title("ABM v4 validation failures define the ABM v5 roadmap", fontsize=15)
        return fig

    def plot_hypothesis_status_table(self, table: pl.DataFrame) -> plt.Figure:
        order = {"supported": 0, "not_supported": 1, "not supported": 1, "diagnostic_only": 2, "diagnostic only": 2}
        rows = sorted(table.to_dicts(), key=lambda row: (order.get(str(row["status"]).lower(), 3), str(row["hypothesis"])))
        fig, ax = plt.subplots(figsize=(11, 6.8))
        ax.axis("off")
        y = 0.90
        current_group = None
        for row in rows:
            group = str(row["status"]).replace("_", " ")
            if group != current_group:
                ax.text(0.04, y, group.upper(), fontsize=10, fontweight="bold", color="#333333")
                y -= 0.055
                current_group = group
            ax.add_patch(Rectangle((0.04, y - 0.030), 0.28, 0.048, facecolor="#f4f4f4", edgecolor="#dddddd"))
            ax.add_patch(Rectangle((0.33, y - 0.030), 0.28, 0.048, facecolor="#ffffff", edgecolor="#dddddd"))
            ax.add_patch(Rectangle((0.62, y - 0.030), 0.34, 0.048, facecolor="#ffffff", edgecolor="#dddddd"))
            ax.text(0.05, y - 0.006, self._wrap(row["hypothesis"], 31), fontsize=7.4, va="center")
            ax.text(0.34, y - 0.006, self._wrap(row["short_evidence"], 35), fontsize=7.4, va="center")
            ax.text(0.63, y - 0.006, self._wrap(row["implication"], 42), fontsize=7.4, va="center")
            y -= 0.058
        ax.set_title("ABM v4 final hypothesis status", fontsize=15)
        return fig

    def write_source_tables(self, tables: dict[str, pl.DataFrame]) -> tuple[Path, ...]:
        """Write narrative source tables."""
        paths = []
        for name in NARRATIVE_SOURCE_TABLE_NAMES:
            path = self.paths.final_tables_narrative / name
            tables[name].write_csv(path)
            paths.append(path)
        return tuple(paths)

    def write_plots(self, plots: dict[str, plt.Figure]) -> tuple[Path, ...]:
        """Write narrative plots as PNG and SVG."""
        paths = []
        for name in NARRATIVE_PLOT_NAMES:
            figure = plots[name]
            for suffix in (".png", ".svg"):
                path = self.paths.final_plots_narrative / f"{name}{suffix}"
                figure.savefig(path, bbox_inches="tight", dpi=180)
                paths.append(path)
            plt.close(figure)
        return tuple(paths)

    def copy_plots_to_output_directory(self, plot_paths: tuple[Path, ...]) -> tuple[Path, ...]:
        """Copy narrative plots to the portfolio output plot folder."""
        self.paths.outputs_plots_abm_v4_final_narrative.mkdir(parents=True, exist_ok=True)
        copied = []
        for path in plot_paths:
            destination = self.paths.outputs_plots_abm_v4_final_narrative / path.name
            copy2(path, destination)
            copied.append(destination)
        return tuple(copied)

    def build_plot_index(
        self,
        source_table_paths: tuple[Path, ...],
        plot_paths: tuple[Path, ...],
        copied_plot_paths: tuple[Path, ...],
    ) -> pl.DataFrame:
        """Build the narrative plot artifact index."""
        rows = []
        for path in source_table_paths:
            rows.append(self._index_row("narrative_source_table", path, "Source table for a Phase 29B narrative plot."))
        for path in plot_paths:
            rows.append(self._index_row("narrative_plot", path, "Narrative-grade ABM v4 final figure."))
        for path in copied_plot_paths:
            rows.append(self._index_row("narrative_plot_copy", path, "Copied plot for portfolio/report assets."))
        for path in (*self.required_input_paths(), *self.optional_input_paths()):
            if path.exists():
                rows.append(self._index_row("input_source", path, "Input used or available to Phase 29B."))
        return pl.DataFrame(rows)

    def _plot_checklist_matrix(self, table: pl.DataFrame, label_col: str, status_col: str, note_col: str, title: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(11, max(4.5, 0.48 * table.height + 1.4)))
        ax.axis("off")
        color_map = {"pass": "#d9ead3", "ready": "#d9ead3", "supported": "#d9ead3", "partial": "#fff2cc", "limited": "#fff2cc", "blocked": "#f4cccc", "fail": "#f4cccc", "not_scenario_ready": "#f4cccc"}
        ax.text(0.05, 0.94, "Check", fontweight="bold")
        ax.text(0.45, 0.94, "Evidence status", fontweight="bold")
        ax.text(0.66, 0.94, "Implication", fontweight="bold")
        for i, row in enumerate(table.to_dicts()):
            y = 0.88 - i * 0.075
            status = str(row[status_col])
            ax.add_patch(Rectangle((0.04, y - 0.028), 0.37, 0.055, facecolor="#ffffff", edgecolor="#dddddd"))
            ax.add_patch(Rectangle((0.43, y - 0.028), 0.18, 0.055, facecolor=color_map.get(status, "#eeeeee"), edgecolor="#dddddd"))
            ax.add_patch(Rectangle((0.63, y - 0.028), 0.33, 0.055, facecolor="#ffffff", edgecolor="#dddddd"))
            ax.text(0.05, y, self._wrap(str(row[label_col]).replace("_", " "), 40), va="center", fontsize=8)
            ax.text(0.52, y, status.replace("_", " "), va="center", ha="center", fontsize=8)
            ax.text(0.64, y, self._wrap(str(row[note_col]), 44), va="center", fontsize=8)
        ax.set_title(title, fontsize=15)
        return fig

    def _mechanism_row(
        self,
        mechanism: str,
        phase_or_branch: str,
        status: str,
        reason: str,
        retained_value: str,
        future_use: str,
        scenario_status: str,
        rejected: pl.DataFrame | None = None,
        source_name: str | None = None,
    ) -> dict[str, str]:
        if rejected is not None and source_name is not None:
            matching = rejected.filter(pl.col("mechanism") == source_name)
            if not matching.is_empty():
                row = matching.row(0, named=True)
                reason = row.get("reason_rejected_or_limited", reason)
                retained_value = row.get("retained_value", retained_value) or retained_value
                future_use = row.get("future_use", future_use) or future_use
                scenario_status = row.get("scenario_status", scenario_status) or scenario_status
        return {
            "mechanism": mechanism,
            "phase_or_branch": phase_or_branch,
            "status": status,
            "reason": reason,
            "retained_value": retained_value,
            "future_use": future_use,
            "scenario_status": scenario_status,
        }

    def _q_check(self, check: str, status: str, evidence: str, implication: str, source_file: str) -> dict[str, str]:
        return {
            "check": check,
            "status": status,
            "quantitative_evidence": evidence,
            "implication": implication,
            "source_file": source_file,
        }

    def _capability_rows_from_optional_table(self, table: pl.DataFrame, source_file: str) -> list[dict[str, object]]:
        columns = set(table.columns)
        if {"capability_type", "atlas_observed", "io_imputed", "unavailable"}.issubset(columns):
            rows = table.select(["capability_type", "atlas_observed", "io_imputed", "unavailable"]).to_dicts()
            for row in rows:
                row["source_file"] = source_file
                row["interpretation"] = "Capability coverage read from optional IO capability diagnostic."
            return rows
        return []

    def _read_optional_csv(self, path: Path) -> pl.DataFrame | None:
        return pl.read_csv(path) if path.exists() else None

    def _first_number(self, row: dict[str, object], names: tuple[str, ...]) -> float | None:
        for name in names:
            value = row.get(name)
            if isinstance(value, (int, float)):
                return abs(float(value))
        return None

    def _diagnostic_error_score(self, value: str) -> float:
        lower = str(value).lower()
        if "supported" in lower or "survive" in lower or "best" in lower:
            return 0.25
        if "diagnostic" in lower or "limited" in lower or "partial" in lower:
            return 0.55
        return 0.85

    def _tradeoff_interpretation(self, variant: str) -> str:
        if variant == "frontier_gap_readiness":
            return "aggregate-safe baseline"
        if variant == "historical_frontier_gap_only":
            return "transition-mechanism benchmark"
        if "EID" in variant or "eid" in variant:
            return "EID variant rejected as ABM v4 behavioural rule"
        return "comparison variant"

    def _q_panel_rows_note(self, quality: pl.DataFrame | None) -> str:
        if quality is None or "rows" not in quality.columns:
            return "108,152 panel rows; 108,130 expected node-years"
        return f"{max(quality['rows'].to_list())} panel rows; broad coverage"

    def _recommendation_note(self, recommendation: pl.DataFrame | None) -> str:
        if recommendation is None or recommendation.is_empty():
            return "recommendation: aggregate_only_energy_mix_usable"
        row = recommendation.row(0, named=True)
        return f"recommendation: {row.get('recommendation', row.get('recommended_next_action', 'aggregate_only_energy_mix_usable'))}"

    def _invalid_share_note(self, quality: pl.DataFrame | None) -> str:
        if quality is None:
            return "invalid share rows present in Phase 27 diagnostics"
        if "negative_count" in quality.columns:
            return f"negative source values observed; total negative count {sum(quality['negative_count'].to_list())}"
        return "invalid share rows present"

    def _negative_count_note(self, quality: pl.DataFrame | None) -> str:
        if quality is None or "negative_count" not in quality.columns:
            return "negative value count unavailable"
        return f"negative value count: {sum(quality['negative_count'].to_list())}"

    def _best_predictor_note(self, screening: pl.DataFrame | None) -> str:
        if screening is None or "abs_correlation" not in screening.columns or screening.is_empty():
            return "best abs predictor correlation unavailable"
        best = screening.sort("abs_correlation", descending=True).row(0, named=True)
        return f"best abs predictor correlation: {best.get('abs_correlation')} ({best.get('predictor')})"

    def _index_row(self, artifact_type: str, path: Path, description: str) -> dict[str, str]:
        return {
            "artifact_type": artifact_type,
            "path": self._relative(path),
            "description": description,
        }

    def _wrap(self, value: str, width: int) -> str:
        return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.paths.project_root)).replace("\\", "/")
        except ValueError:
            return str(path).replace("\\", "/")


ABMV4PolishedPlotBuilder.__bases__ = (ABMV4NarrativePlotBuilder,)
