from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
