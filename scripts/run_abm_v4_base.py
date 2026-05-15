from __future__ import annotations

import argparse

import polars as pl

from src.abm_v4.capabilities import CapabilityUpdater, IOCapabilityBuilder
from src.abm_v4.config import ABMV4Config
from src.abm_v4.diagnostics import build_path_audit_rows, format_path_audit_table
from src.abm_v4.ecosystem import EcosystemMapper
from src.abm_v4.emissions import (
    EmissionsTransitionCalibrator,
    EmissionsTransitionHypothesisDiagnostics,
    EmissionsTransitionVariantComparator,
    EmissionsUpdater,
    HISTORICAL_FRONTIER_GAP_EID_DIAGNOSTIC_MODE,
)
from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.production import ProductionFeasibilityEngine
from src.abm_v4.reporting import ABMV4FinalArtifactBuilder
from src.abm_v4.simulation import (
    MultiYearBaseSimulator,
    inspect_base_model_readiness,
    run_one_step_base_orchestration,
)
from src.abm_v4.state import build_state_panel, repair_capability_coverage
from src.abm_v4.suppliers import SupplierNetworkBuilder
from src.abm_v4.validation import MultiYearHistoricalValidator
from src.abm_v4.validation import (
    ABMV4FinalConsolidator,
    ElectricityDataAudit,
    ElectricityTransitionRegimeDiagnostics,
    EssentialInputDampenerTester,
    EssentialInputDependenceDiagnostics,
    EssentialInputFailureModeDiagnostics,
    AdaptiveEIDCalibrationDiagnostics,
    HighEmissionsDampeningDiagnostics,
    MultiYearEIDDiagnosticValidator,
    QEnergyMixAudit,
    RawEoraElectricityDataAudit,
    StructuralSignatureDiagnostics,
    TransitionRuleTradeoffDiagnostics,
    build_multiyear_base_model_comparison,
    write_multiyear_base_model_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ABM v4 base readiness and dry-run audit entry point."
    )
    parser.add_argument("--start-year", type=int, default=1995)
    parser.add_argument("--end-year", type=int, default=2016)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit source paths without creating ABM v4 output folders.",
    )
    parser.add_argument(
        "--create-output-dirs",
        action="store_true",
        help="Create data/abm_v4 output directories for a future run.",
    )
    parser.add_argument(
        "--build-state",
        action="store_true",
        help="Build the ABM v4 state panel. Requires --create-output-dirs to write files.",
    )
    parser.add_argument(
        "--build-ecosystems",
        action="store_true",
        help="Assign productive ecosystems to the ABM v4 state panel.",
    )
    parser.add_argument(
        "--build-supplier-edges",
        action="store_true",
        help="Build observed historical supplier-buyer edges.",
    )
    parser.add_argument(
        "--build-raw-t-supplier-edges",
        action="store_true",
        help="Build raw Eora T supplier-buyer edges and compare with legacy edges.",
    )
    parser.add_argument(
        "--build-supplier-candidate-base",
        action="store_true",
        help="Build compact Phase 4B-prep supplier candidate tables.",
    )
    parser.add_argument(
        "--build-supplier-opportunities",
        action="store_true",
        help="Build Phase 4B supplier opportunity sets from compact candidate tables.",
    )
    parser.add_argument(
        "--build-supplier-rewiring",
        action="store_true",
        help="Build Phase 4C initial weights, rewiring flags, and updated weights.",
    )
    parser.add_argument(
        "--build-capability-update",
        action="store_true",
        help="Build Phase 5 one-step capability exposure and update panels.",
    )
    parser.add_argument(
        "--build-production-feasibility",
        action="store_true",
        help="Build Phase 6 one-step production feasibility diagnostics.",
    )
    parser.add_argument(
        "--build-emissions-update",
        action="store_true",
        help="Build Phase 7 one-step emissions intensity update and decomposition.",
    )
    parser.add_argument(
        "--emissions-transition-mode",
        choices=("frontier_gap_readiness", "legacy_raw_log", "historical_frontier_gap_only", HISTORICAL_FRONTIER_GAP_EID_DIAGNOSTIC_MODE),
        default=None,
        help="Emissions transition rule to use for --build-emissions-update.",
    )
    parser.add_argument(
        "--emissions-parameter-file",
        default=None,
        help="Optional JSON parameter file for calibrated-historical emissions rules.",
    )
    parser.add_argument(
        "--build-emissions-transition-comparison",
        action="store_true",
        help="Compare frontier-gap readiness emissions transition with legacy raw-log mode.",
    )
    parser.add_argument(
        "--run-one-step-base",
        action="store_true",
        help="Run Phase 8 one-step ABM v4 base orchestration and validation.",
    )
    parser.add_argument(
        "--run-multiyear-base",
        action="store_true",
        help="Run Phase 10 historical multi-year ABM v4 base simulation.",
    )
    parser.add_argument(
        "--run-multiyear-EID-diagnostic",
        action="store_true",
        help="Run Phase 25 EID diagnostic multi-year integration audit.",
    )
    parser.add_argument(
        "--validate-multiyear-base",
        action="store_true",
        help="Build Phase 11 historical validation and calibration diagnostics for the multi-year base run.",
    )
    parser.add_argument(
        "--calibrate-emissions-transition",
        action="store_true",
        help="Build Phase 12 emissions-transition parameter calibration diagnostics.",
    )
    parser.add_argument(
        "--diagnose-emissions-hypotheses",
        action="store_true",
        help="Build Phase 13 hypothesis diagnostics for weak emissions-transition calibration.",
    )
    parser.add_argument(
        "--compare-emissions-transition-variants",
        action="store_true",
        help="Build Phase 14 theory-structured emissions-transition variant diagnostics.",
    )
    parser.add_argument(
        "--compare-multiyear-base-models",
        action="store_true",
        help="Compare available default and calibrated-historical multi-year base outputs.",
    )
    parser.add_argument(
        "--diagnose-transition-rule-tradeoffs",
        action="store_true",
        help="Build Phase 16 transition-rule error decomposition and sign-failure diagnostics.",
    )
    parser.add_argument(
        "--diagnose-high-emissions-dampening",
        action="store_true",
        help="Build Phase 17 high-emissions node and readiness-dampening diagnostics.",
    )
    parser.add_argument(
        "--audit-electricity-data",
        action="store_true",
        help="Build Phase 18 electricity and China EI data audit diagnostics.",
    )
    parser.add_argument(
        "--audit-raw-eora-electricity-data",
        action="store_true",
        help="Build Phase 19 raw Eora-derived electricity data path audit diagnostics.",
    )
    parser.add_argument(
        "--diagnose-electricity-transition-regime",
        action="store_true",
        help="Build Phase 20 electricity-specific transition regime diagnostics.",
    )
    parser.add_argument(
        "--diagnose-structural-signatures",
        action="store_true",
        help="Build Phase 21 structural-signature discovery and transition-inertia proxy diagnostics.",
    )
    parser.add_argument(
        "--diagnose-essential-input-dependence",
        action="store_true",
        help="Build Phase 22 essential-input and IO structural-dependence diagnostics.",
    )
    parser.add_argument(
        "--test-essential-input-dampener",
        action="store_true",
        help="Build Phase 23 essential-input dampener candidate validation diagnostics.",
    )
    parser.add_argument(
        "--diagnose-eid-failure-modes",
        action="store_true",
        help="Build Phase 24 high-EID dampener failure-mode and heterogeneity diagnostics.",
    )
    parser.add_argument(
        "--diagnose-adaptive-EID-calibration",
        action="store_true",
        help="Build Phase 26 adaptive EID calibration diagnostics.",
    )
    parser.add_argument(
        "--audit-q-energy-mix",
        action="store_true",
        help="Build Phase 27 Eora Q energy-mix audit and transition-error diagnostics.",
    )
    parser.add_argument(
        "--finalize-abm-v4",
        action="store_true",
        help="Build Phase 28 final ABM v4 consolidation and two-rule validation outputs.",
    )
    parser.add_argument(
        "--build-final-abm-v4-plots-tables",
        action="store_true",
        help="Build Phase 29A final ABM v4 plots, clean tables, and artifact index.",
    )
    parser.add_argument("--calibration-random-search-iterations", type=int, default=200)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--calibration-train-end-year", type=int, default=2011)
    parser.add_argument("--calibration-validation-start-year", type=int, default=2012)
    parser.add_argument("--transition-variant-random-search-iterations", type=int, default=100)
    parser.add_argument("--transition-variant-seed", type=int, default=42)
    parser.add_argument(
        "--transition-variant-target",
        action="append",
        choices=("one_year_rEI", "smoothed_one_year_rEI", "three_year_rEI"),
        default=None,
        help="Target horizon to include in Phase 14. May be supplied multiple times.",
    )
    parser.add_argument(
        "--historical-production-forcing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use observed historical output as the production scale anchor in the base simulation.",
    )
    parser.add_argument(
        "--repair-capability-coverage",
        action="store_true",
        help="Repair ABM v4 state capability coverage by joining Atlas capability data.",
    )
    parser.add_argument(
        "--build-io-capability-model",
        action="store_true",
        help="Build Phase 9C source-aware Atlas/IO capability model fields.",
    )
    parser.add_argument(
        "--audit-io-capability-robustness",
        action="store_true",
        help="Build Phase 9D IO capability robustness and downstream proxy diagnostics.",
    )
    parser.add_argument(
        "--force-rebuild-raw-t-edges",
        action="store_true",
        help="Allow a one-step base run to rebuild raw T edges if that path is implemented.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing compatible ABM v4 component outputs during one-step orchestration.",
    )
    parser.add_argument(
        "--candidate-debug-buyers",
        type=int,
        default=None,
        help="Limit supplier candidate building to the first N buyers for debugging.",
    )
    parser.add_argument(
        "--candidate-debug-years",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Limit historical raw-T candidate aggregation to an inclusive year range.",
    )
    return parser


def load_legacy_edges_for_comparison(
    builder: SupplierNetworkBuilder,
    paths: ABMV4Paths,
) -> pl.DataFrame:
    """Load or build the legacy embodied-emissions edge panel for comparison only."""
    if paths.historical_supplier_edges_path.exists():
        return pl.read_parquet(paths.historical_supplier_edges_path)

    legacy_path = paths.data_abm_legacy / "edges_panel.parquet"
    if not legacy_path.exists():
        raise FileNotFoundError(f"Legacy supplier edge panel not found: {legacy_path}")

    state_panel = builder.load_state_panel()
    legacy_source_edges, _ = builder.normalize_edge_schema(pl.read_parquet(legacy_path))
    legacy_edges = builder.attach_supplier_buyer_metadata(legacy_source_edges, state_panel)
    legacy_edges = builder.compute_historical_ties(legacy_edges)
    return legacy_edges.with_columns(
        pl.lit(str(legacy_path)).alias("source_file"),
        pl.lit("legacy_abm_edges_embodied_emissions").alias("source_type"),
    )


def main() -> None:
    """Inspect ABM v4 base-model readiness without generating simulation data."""
    args = build_parser().parse_args()
    config = ABMV4Config(start_year=args.start_year, end_year=args.end_year)
    paths = ABMV4Paths()

    if args.dry_run:
        rows = build_path_audit_rows(paths, config.start_year, config.end_year)
        print(format_path_audit_table(rows))
        return

    if args.build_state:
        if not args.create_output_dirs:
            raise SystemExit("--build-state requires --create-output-dirs to write outputs.")
        result = build_state_panel(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
            write_outputs=True,
            epsilon=config.epsilon,
        )
        print(f"Selected state source: {result.selected_source}")
        print(f"State panel rows: {result.state_panel.height}")
        print(f"State panel path: {result.output_path}")
        if not args.build_ecosystems:
            return

    if args.build_ecosystems:
        if not args.create_output_dirs:
            raise SystemExit("--build-ecosystems requires --create-output-dirs to write outputs.")
        state_panel_path = paths.state_panel_path(config.start_year, config.end_year)
        if not state_panel_path.exists():
            raise SystemExit(
                "ABM v4 state panel is missing. Run with --build-state --build-ecosystems "
                "--create-output-dirs, or build the state panel first."
            )
        state_panel = pl.read_parquet(state_panel_path)
        eta_ecosystem_adjacent = getattr(config, "eta_ecosystem_adjacent", 0.35)
        mapper = EcosystemMapper(paths=paths, eta_ecosystem_adjacent=eta_ecosystem_adjacent)
        ecosystem_result = mapper.assign_ecosystems(state_panel)
        mapper.write_outputs(ecosystem_result, state_panel_path=state_panel_path)
        report_row = ecosystem_result.assignment_report.to_dicts()[0]
        print(f"Ecosystem source: {ecosystem_result.ecosystem_source}")
        print(f"Mapped nodes: {report_row['mapped_nodes']}")
        print(f"Unmapped nodes: {report_row['unmapped_nodes']}")
        print(f"Ecosystem mapping path: {paths.ecosystem_mapping_path}")
        return

    if args.repair_capability_coverage:
        if not args.create_output_dirs:
            raise SystemExit(
                "--repair-capability-coverage requires --create-output-dirs to write outputs."
            )
        result = repair_capability_coverage(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
            write_outputs=True,
        )
        report_row = result.join_report.to_dicts()[0]
        print("Repaired ABM v4 capability coverage.")
        print(f"Capability source: {report_row['source_file']}")
        print(f"Join keys: {report_row['selected_join_keys']}")
        print(f"Matched share: {report_row['matched_share']}")
        print(
            "General capability fill share: "
            f"{report_row['general_capability_fill_share_before']} -> "
            f"{report_row['general_capability_fill_share_after']}"
        )
        print(
            "Green capability fill share: "
            f"{report_row['green_capability_fill_share_before']} -> "
            f"{report_row['green_capability_fill_share_after']}"
        )
        print(f"Updated state panel path: {paths.state_panel_path(config.start_year, config.end_year)}")
        print(f"Capability join report path: {paths.capability_join_report_path}")
        return

    if args.build_io_capability_model:
        if not args.create_output_dirs:
            raise SystemExit(
                "--build-io-capability-model requires --create-output-dirs to write outputs."
            )
        builder = IOCapabilityBuilder(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
            config=config.capability,
        )
        result = builder.build_io_capability_model()
        builder.write_outputs(result)
        report_row = result.model_report.to_dicts()[0]
        print("Built IO-derived capability model.")
        print(f"Selected year: {report_row['selected_year']}")
        print(f"General IO-imputed count: {report_row['io_imputed_general_count']}")
        print(f"Green IO-imputed count: {report_row['io_imputed_green_count']}")
        print(f"Selected lambda general up: {report_row['selected_lambda_general_up']}")
        print(f"Selected lambda green up: {report_row['selected_lambda_green_up']}")
        print(f"IO capability model report path: {paths.io_capability_model_report_path}")
        return

    if args.audit_io_capability_robustness:
        if not args.create_output_dirs:
            raise SystemExit(
                "--audit-io-capability-robustness requires --create-output-dirs to write diagnostics."
            )
        builder = IOCapabilityBuilder(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
            config=config.capability,
        )
        result = builder.build_io_capability_robustness()
        builder.write_robustness_outputs(result)
        print("Built IO capability robustness diagnostics.")
        print(f"Robustness path: {paths.io_capability_robustness_path}")
        print(f"Threshold sensitivity path: {paths.io_capability_threshold_sensitivity_path}")
        print(f"Downstream audit path: {paths.io_downstream_exposure_audit_path}")
        return

    if args.build_supplier_edges:
        if not args.create_output_dirs:
            raise SystemExit("--build-supplier-edges requires --create-output-dirs to write outputs.")
        builder = SupplierNetworkBuilder(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
        )
        result = builder.build_historical_edges()
        written = builder.write_historical_edges(result)
        report_row = written.edge_report.to_dicts()[0]
        print(f"Selected edge source: {written.selected_source.path}")
        print(f"Source type: {written.selected_source.source_type}")
        print(f"Historical edge rows: {report_row['row_count']}")
        print(f"Unique supplier-buyer pairs: {report_row['unique_supplier_buyer_pairs']}")
        print(f"Historical supplier edges path: {written.output_path}")
        return

    if args.build_raw_t_supplier_edges:
        if not args.create_output_dirs:
            raise SystemExit(
                "--build-raw-t-supplier-edges requires --create-output-dirs to write outputs."
            )
        builder = SupplierNetworkBuilder(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
        )
        legacy_edges = load_legacy_edges_for_comparison(builder, paths)
        raw_report, _ = builder.build_and_write_raw_t_supplier_edges(
            years=range(config.start_year, config.end_year + 1),
            legacy_edges=legacy_edges,
        )
        raw_edges_path = paths.raw_t_supplier_edges_path
        comparison_path = paths.supplier_edge_source_comparison_path
        report_row = raw_report.to_dicts()[0]
        print("Selected edge source: raw Eora T matrices")
        print("Source type: raw_eora_T")
        print(f"Raw T edge rows: {report_row['row_count']}")
        print(f"Unique supplier-buyer pairs: {report_row['unique_supplier_buyer_pairs']}")
        print(f"Raw T supplier edges path: {raw_edges_path}")
        print(f"Edge source comparison path: {comparison_path}")
        return

    if args.build_supplier_candidate_base:
        if not args.create_output_dirs:
            raise SystemExit(
                "--build-supplier-candidate-base requires --create-output-dirs to write outputs."
            )
        builder = SupplierNetworkBuilder(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
        )
        debug_years = (
            tuple(args.candidate_debug_years)
            if args.candidate_debug_years is not None
            else None
        )
        historical_candidates = builder.build_historical_top_supplier_candidates(
            max_historical_suppliers_per_buyer=25,
            debug_buyers=args.candidate_debug_buyers,
            debug_years=debug_years,
        )
        same_sector_candidates = builder.build_same_sector_supplier_pool(
            max_same_sector_candidates_per_buyer=25,
            debug_buyers=args.candidate_debug_buyers,
        )
        ecosystem_candidates = builder.build_ecosystem_supplier_pool(
            max_ecosystem_candidates_per_buyer=25,
            historical_candidates=historical_candidates,
            same_sector_candidates=same_sector_candidates,
            debug_buyers=args.candidate_debug_buyers,
        )
        report = builder.build_supplier_candidate_base_report(
            historical_candidates=historical_candidates,
            same_sector_candidates=same_sector_candidates,
            ecosystem_candidates=ecosystem_candidates,
        )
        builder.write_supplier_candidate_base(
            historical_candidates=historical_candidates,
            same_sector_candidates=same_sector_candidates,
            ecosystem_candidates=ecosystem_candidates,
            report=report,
        )
        report_row = report.to_dicts()[0]
        print("Built compact supplier candidate base.")
        print(f"Historical candidate rows: {report_row['historical_candidate_rows']}")
        print(f"Same-sector candidate rows: {report_row['same_sector_candidate_rows']}")
        print(f"Ecosystem candidate rows: {report_row['ecosystem_candidate_rows']}")
        print(f"Historical candidates path: {paths.supplier_candidates_historical_top_path}")
        print(f"Same-sector pool path: {paths.supplier_pool_same_sector_path}")
        print(f"Ecosystem pool path: {paths.supplier_pool_ecosystem_path}")
        print(f"Candidate base report path: {paths.supplier_candidate_base_report_path}")
        return

    if args.build_supplier_opportunities:
        if not args.create_output_dirs:
            raise SystemExit(
                "--build-supplier-opportunities requires --create-output-dirs to write outputs."
            )
        builder = SupplierNetworkBuilder(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
        )
        opportunities = builder.build_supplier_opportunity_sets(
            supplier_friction=config.supplier_friction,
            supplier_choice=config.supplier_choice,
            beta_supplier_choice=getattr(config.supplier_choice, "beta_supplier_choice", 1.0),
            epsilon=config.epsilon,
        )
        report = builder.build_opportunity_set_report(opportunities)
        builder.write_supplier_opportunity_sets(opportunities, report)
        report_row = report.to_dicts()[0]
        print("Built supplier opportunity sets.")
        print(f"Opportunity rows: {report_row['opportunity_rows']}")
        print(f"Median candidates per buyer: {report_row['median_candidates_per_buyer']}")
        print(f"Buyers with probability sum error: {report_row['buyers_with_probability_sum_error']}")
        print(f"Opportunity set path: {paths.supplier_opportunity_sets_path}")
        print(f"Opportunity set report path: {paths.supplier_opportunity_set_report_path}")
        return

    if args.build_supplier_rewiring:
        if not args.create_output_dirs:
            raise SystemExit(
                "--build-supplier-rewiring requires --create-output-dirs to write outputs."
            )
        builder = SupplierNetworkBuilder(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
        )
        random_seed = getattr(config, "random_seed", 42)
        initial_weights, rewiring_flags, updated_weights, report = (
            builder.build_supplier_rewiring_outputs(
                supplier_choice=config.supplier_choice,
                random_seed=random_seed,
            )
        )
        builder.write_supplier_rewiring_outputs(
            initial_weights=initial_weights,
            rewiring_flags=rewiring_flags,
            updated_weights=updated_weights,
            report=report,
        )
        report_row = report.to_dicts()[0]
        print("Built supplier rewiring weights.")
        print(f"Number of buyers: {report_row['number_of_buyers']}")
        print(f"Rewired buyer share: {report_row['rewired_buyer_share']}")
        print(f"Max updated weight sum error: {report_row['max_updated_weight_sum_error']}")
        print(f"Initial weights path: {paths.supplier_initial_weights_path}")
        print(f"Rewiring flags path: {paths.supplier_rewiring_flags_path}")
        print(f"Updated weights path: {paths.supplier_updated_weights_path}")
        print(f"Supplier rewiring report path: {paths.supplier_rewiring_report_path}")
        return

    if args.build_capability_update:
        if not args.create_output_dirs:
            raise SystemExit(
                "--build-capability-update requires --create-output-dirs to write outputs."
            )
        updater = CapabilityUpdater(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
            config=config.capability,
        )
        exposure_panel, update_panel, report = updater.build_capability_update()
        updater.write_outputs(exposure_panel, update_panel, report)
        report_row = report.to_dicts()[0]
        print("Built capability update.")
        print(f"Year: {report_row['year']}")
        print(f"Node count: {report_row['node_count']}")
        print(f"Mean delta cap: {report_row['mean_delta_cap']}")
        print(f"Mean delta gcap: {report_row['mean_delta_gcap']}")
        print(f"Capability exposure path: {paths.capability_exposure_panel_path}")
        print(f"Capability update path: {paths.capability_update_panel_path}")
        print(f"Capability report path: {paths.capability_update_report_path}")
        return

    if args.build_production_feasibility:
        if not args.create_output_dirs:
            raise SystemExit(
                "--build-production-feasibility requires --create-output-dirs to write outputs."
            )
        engine = ProductionFeasibilityEngine(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
            epsilon=config.epsilon,
        )
        panel = engine.build_feasibility_panel()
        report = engine.build_production_feasibility_report(panel)
        engine.write_outputs(panel, report)
        report_row = report.to_dicts()[0]
        print("Built production feasibility diagnostics.")
        print(f"Year: {report_row['year']}")
        print(f"Node count: {report_row['node_count']}")
        print(f"Aggregate feasibility ratio: {report_row['aggregate_feasibility_ratio']}")
        print(f"Constrained node share: {report_row['share_nodes_with_input_feasibility_below_1']}")
        print(f"Production feasibility path: {paths.production_feasibility_panel_path}")
        print(f"Production feasibility report path: {paths.production_feasibility_report_path}")
        return

    if args.build_emissions_update:
        if not args.create_output_dirs:
            raise SystemExit(
                "--build-emissions-update requires --create-output-dirs to write outputs."
            )
        updater = EmissionsUpdater(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
            config=config.emissions,
            transition_mode=args.emissions_transition_mode,
        )
        (
            panel,
            report,
            decomposition,
            historical_summary,
            sector_background,
            frontier_gap_report,
        ) = updater.build_emissions_update(parameter_file=args.emissions_parameter_file)
        updater.write_outputs(
            panel,
            report,
            decomposition,
            historical_summary,
            sector_background,
            frontier_gap_report,
        )
        report_row = report.to_dicts()[0]
        print("Built emissions update.")
        print(f"Transition mode: {report_row['emissions_transition_mode']}")
        print(f"Year: {report_row['year']}")
        print(f"Valid EI nodes: {report_row['valid_EI_nodes']}")
        print(f"Invalid EI nodes: {report_row['invalid_EI_nodes']}")
        print(f"Mean rEI used: {report_row['mean_rEI_used']}")
        print(f"Aggregate delta emissions: {report_row['aggregate_delta_emissions']}")
        print(f"Bad transition flag: {report_row['bad_transition_flag']}")
        print(f"Emissions update path: {paths.emissions_update_panel_path}")
        print(f"Emissions update report path: {paths.emissions_update_report_path}")
        print(f"Emissions decomposition path: {paths.emissions_decomposition_base_path}")
        return

    if args.build_emissions_transition_comparison:
        if not args.create_output_dirs:
            raise SystemExit(
                "--build-emissions-transition-comparison requires --create-output-dirs to write outputs."
            )
        updater = EmissionsUpdater(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
            config=config.emissions,
        )
        comparison = updater.build_transition_comparison()
        updater.write_transition_comparison(comparison)
        print("Built emissions transition comparison.")
        print(f"Comparison path: {paths.emissions_transition_comparison_path}")
        for row in comparison.to_dicts():
            print(
                f"{row['mode']}: mean rEI={row['mean_rEI_used']}, "
                f"delta emissions={row['aggregate_delta_emissions']}"
            )
        return

    if args.run_one_step_base:
        if not args.create_output_dirs:
            raise SystemExit(
                "--run-one-step-base requires --create-output-dirs to write validation outputs."
            )
        result = run_one_step_base_orchestration(
            paths=paths,
            config=config,
            reuse_existing=True if args.reuse_existing else True,
            force_rebuild_raw_t_edges=args.force_rebuild_raw_t_edges,
            write_outputs=True,
        )
        status = result.validation.status
        print("Ran one-step ABM v4 base validation.")
        print(f"Overall status: {status['overall_status']}")
        print(f"Warning layers: {', '.join(status['warning_layers']) if status['warning_layers'] else '-'}")
        print(f"Failed layers: {', '.join(status['failed_layers']) if status['failed_layers'] else '-'}")
        print(f"Validation CSV: {paths.one_step_base_validation_report_csv_path}")
        print(f"Validation Markdown: {paths.one_step_base_validation_report_md_path}")
        print(f"Status JSON: {paths.one_step_base_status_json_path}")
        return

    if args.run_multiyear_base:
        if not args.create_output_dirs:
            raise SystemExit(
                "--run-multiyear-base requires --create-output-dirs to write simulation outputs."
            )
        simulator = MultiYearBaseSimulator(
            paths=paths,
            config=config,
            historical_production_forcing=args.historical_production_forcing,
            reuse_existing=args.reuse_existing,
            emissions_transition_mode=args.emissions_transition_mode,
            emissions_parameter_file=args.emissions_parameter_file,
        )
        result = simulator.run()
        simulator.write_outputs(result)
        validation_row = result.validation_report.to_dicts()[0]
        print("Ran ABM v4 multi-year base simulation.")
        print(f"Years: {validation_row['simulation_start_year']}-{validation_row['simulation_end_year']}")
        print(f"Status: {validation_row['status']}")
        print(f"Historical production forcing: {validation_row['historical_production_forcing']}")
        if args.emissions_transition_mode == "historical_frontier_gap_only":
            print(f"State panel: {paths.base_multiyear_state_panel_historical_frontier_gap_path}")
            print(f"Summary panel: {paths.base_multiyear_summary_panel_historical_frontier_gap_path}")
            print(
                "Validation report: "
                f"{paths.base_multiyear_validation_report_historical_frontier_gap_csv_path}"
            )
        else:
            print(f"State panel: {paths.base_multiyear_state_panel_path}")
            print(f"Summary panel: {paths.base_multiyear_summary_panel_path}")
            print(f"Validation report: {paths.base_multiyear_validation_report_path}")
        return

    if args.run_multiyear_EID_diagnostic:
        if not args.create_output_dirs:
            raise SystemExit(
                "--run-multiyear-EID-diagnostic requires --create-output-dirs to write diagnostic outputs."
            )
        simulator = MultiYearBaseSimulator(
            paths=paths,
            config=config,
            historical_production_forcing=args.historical_production_forcing,
            reuse_existing=args.reuse_existing,
            emissions_transition_mode=HISTORICAL_FRONTIER_GAP_EID_DIAGNOSTIC_MODE,
            emissions_parameter_file=args.emissions_parameter_file,
        )
        result = simulator.run()
        simulator.write_outputs(result)
        validator = MultiYearEIDDiagnosticValidator(paths=paths)
        validation = validator.run()
        validator.write_outputs(validation)
        recommendation = validation.recommendation.to_dicts()[0]
        validation_row = result.validation_report.to_dicts()[0]
        print("Ran ABM v4 EID diagnostic multi-year integration audit.")
        print(f"Years: {validation_row['simulation_start_year']}-{validation_row['simulation_end_year']}")
        print(f"Status: {validation_row['status']}")
        print(f"Recommendation: {recommendation['recommendation']}")
        print(f"State panel: {paths.base_multiyear_state_panel_EID_diagnostic_path}")
        print(f"Validation report: {paths.multiyear_EID_diagnostic_report_path}")
        return

    if args.validate_multiyear_base:
        if not args.create_output_dirs:
            raise SystemExit(
                "--validate-multiyear-base requires --create-output-dirs to write validation outputs."
            )
        validator = MultiYearHistoricalValidator(paths=paths, config=config)
        result = validator.build()
        validator.write_outputs(result)
        latest = result.error_summary.sort("year").tail(1).to_dicts()[0]
        print("Built ABM v4 multi-year historical validation diagnostics.")
        print(f"Latest year: {latest['year']}")
        print(f"Latest aggregate emissions pct error: {latest['aggregate_emissions_pct_error']}")
        print(f"Error panel: {paths.multiyear_error_panel_path}")
        print(f"Error summary: {paths.multiyear_error_summary_path}")
        print(f"Markdown report: {paths.multiyear_validation_report_md_path}")
        return

    if args.calibrate_emissions_transition:
        if not args.create_output_dirs:
            raise SystemExit(
                "--calibrate-emissions-transition requires --create-output-dirs to write validation outputs."
            )
        calibrator = EmissionsTransitionCalibrator(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
            config=config.emissions,
            random_search_iterations=args.calibration_random_search_iterations,
            seed=args.calibration_seed,
            train_end_year=args.calibration_train_end_year,
            validation_start_year=args.calibration_validation_start_year,
        )
        result = calibrator.run()
        calibrator.write_outputs(result)
        validation = result.validation_summary.filter(pl.col("split") == "validation").to_dicts()[0]
        print("Built ABM v4 emissions-transition calibration diagnostics.")
        print(f"Calibration rows: {result.dataset.height}")
        print(f"Validation MAE: {validation['mae']}")
        print(f"Validation bias: {validation['bias']}")
        print(f"Best parameters: {paths.emissions_best_parameters_path}")
        print(f"Calibration report: {paths.emissions_calibration_report_path}")
        return

    if args.diagnose_emissions_hypotheses:
        if not args.create_output_dirs:
            raise SystemExit(
                "--diagnose-emissions-hypotheses requires --create-output-dirs to write validation outputs."
            )
        diagnostics = EmissionsTransitionHypothesisDiagnostics(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
            config=config.emissions,
        )
        result = diagnostics.run()
        diagnostics.write_outputs(result)
        print("Built ABM v4 emissions-transition hypothesis diagnostics.")
        print(f"Hypotheses tested: {result.hypothesis_diagnosis.height}")
        print(f"Diagnosis table: {paths.emissions_hypothesis_diagnosis_path}")
        print(f"Diagnostic report: {paths.emissions_hypothesis_diagnostic_report_path}")
        return

    if args.compare_emissions_transition_variants:
        if not args.create_output_dirs:
            raise SystemExit(
                "--compare-emissions-transition-variants requires --create-output-dirs to write validation outputs."
            )
        comparator = EmissionsTransitionVariantComparator(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
            config=config.emissions,
            random_search_iterations=args.transition_variant_random_search_iterations,
            seed=args.transition_variant_seed,
            targets=args.transition_variant_target,
        )
        result = comparator.run()
        comparator.write_outputs(result)
        recommendation = result.recommendation.to_dicts()[0]
        print("Built ABM v4 emissions-transition variant comparison diagnostics.")
        print(f"Variants compared: {result.results.height}")
        print(f"Recommended variant: {recommendation['recommended_model_variant']}")
        print(f"Recommended target: {recommendation['recommended_target']}")
        print(f"Recommended frontier: {recommendation['recommended_frontier']}")
        print(f"Results table: {paths.emissions_transition_variant_results_path}")
        print(f"Recommendation table: {paths.emissions_transition_variant_recommendation_path}")
        print(f"Variant report: {paths.emissions_transition_variant_report_path}")
        return

    if args.compare_multiyear_base_models:
        if not args.create_output_dirs:
            raise SystemExit(
                "--compare-multiyear-base-models requires --create-output-dirs to write validation outputs."
            )
        comparison, markdown = build_multiyear_base_model_comparison(paths)
        write_multiyear_base_model_comparison(paths, comparison, markdown)
        print("Built ABM v4 multi-year base model comparison.")
        print(f"Compared variants: {comparison.height}")
        print(f"Comparison CSV: {paths.multiyear_base_model_comparison_csv_path}")
        print(f"Comparison Markdown: {paths.multiyear_base_model_comparison_md_path}")
        return

    if args.diagnose_transition_rule_tradeoffs:
        if not args.create_output_dirs:
            raise SystemExit(
                "--diagnose-transition-rule-tradeoffs requires --create-output-dirs to write validation outputs."
            )
        diagnostics = TransitionRuleTradeoffDiagnostics(paths)
        result = diagnostics.run()
        diagnostics.write_outputs(result)
        print("Built ABM v4 transition-rule tradeoff diagnostics.")
        print(f"Comparison rows: {result.sign_failure_panel.height}")
        print(f"Hypotheses tested: {result.hypothesis_tests.height}")
        print(f"Tradeoff report: {paths.transition_rule_error_tradeoff_report_path}")
        return

    if args.diagnose_high_emissions_dampening:
        if not args.create_output_dirs:
            raise SystemExit(
                "--diagnose-high-emissions-dampening requires --create-output-dirs to write validation outputs."
            )
        diagnostics = HighEmissionsDampeningDiagnostics(paths)
        result = diagnostics.run()
        diagnostics.write_outputs(result)
        recommendation = result.recommendation.to_dicts()[0]
        print("Built ABM v4 high-emissions dampening diagnostics.")
        print(f"Recommendation: {recommendation['recommendation']}")
        print(f"Concentration rows: {result.concentration.height}")
        print(f"Phase 17 report: {paths.phase17_high_emissions_dampening_report_path}")
        return

    if args.audit_electricity_data:
        if not args.create_output_dirs:
            raise SystemExit("--audit-electricity-data requires --create-output-dirs to write validation outputs.")
        audit = ElectricityDataAudit(paths=paths, start_year=config.start_year, end_year=config.end_year)
        result = audit.run()
        audit.write_outputs(result)
        recommendation = result.recommendation.to_dicts()[0]
        print("Built ABM v4 electricity and China EI data audit.")
        print(f"Recommendation: {recommendation['recommended_next_action']}")
        print(f"Electricity nodes: {result.inventory.height}")
        print(f"Phase 18 report: {paths.electricity_data_audit_report_path}")
        return

    if args.audit_raw_eora_electricity_data:
        if not args.create_output_dirs:
            raise SystemExit(
                "--audit-raw-eora-electricity-data requires --create-output-dirs to write validation outputs."
            )
        audit = RawEoraElectricityDataAudit(paths=paths, start_year=config.start_year, end_year=config.end_year)
        result = audit.run()
        audit.write_outputs(result)
        recommendation = result.recommendation.to_dicts()[0]
        print("Built ABM v4 raw Eora-derived electricity data path audit.")
        print(f"Recommendation: {recommendation['recommended_next_action']}")
        print(f"Usable sources: {result.source_inventory.filter(pl.col('usable_for_china_electricity')).height}")
        print(f"Phase 19 report: {paths.raw_eora_electricity_data_audit_report_path}")
        return

    if args.diagnose_electricity_transition_regime:
        if not args.create_output_dirs:
            raise SystemExit(
                "--diagnose-electricity-transition-regime requires --create-output-dirs to write validation outputs."
            )
        diagnostics = ElectricityTransitionRegimeDiagnostics(paths=paths, start_year=config.start_year, end_year=config.end_year)
        result = diagnostics.run()
        diagnostics.write_outputs(result)
        recommendation = result.recommendation.to_dicts()[0]
        print("Built ABM v4 electricity transition regime diagnostics.")
        print(f"Recommendation: {recommendation['recommendation']}")
        print(f"Rules compared: {result.rule_comparison.height}")
        print(f"Phase 20 report: {paths.electricity_transition_regime_report_path}")
        return

    if args.diagnose_structural_signatures:
        if not args.create_output_dirs:
            raise SystemExit(
                "--diagnose-structural-signatures requires --create-output-dirs to write validation outputs."
            )
        diagnostics = StructuralSignatureDiagnostics(paths=paths, start_year=config.start_year, end_year=config.end_year)
        result = diagnostics.run()
        diagnostics.write_outputs(result)
        recommendation = result.recommendation.to_dicts()[0]
        print("Built ABM v4 structural-signature diagnostics.")
        print(f"Recommendation: {recommendation['recommendation']}")
        print(f"Node-year rows: {result.node_year_panel.height}")
        print(f"Phase 21 report: {paths.structural_signature_report_path}")
        return

    if args.diagnose_essential_input_dependence:
        if not args.create_output_dirs:
            raise SystemExit(
                "--diagnose-essential-input-dependence requires --create-output-dirs to write validation outputs."
            )
        diagnostics = EssentialInputDependenceDiagnostics(paths=paths, start_year=config.start_year, end_year=config.end_year)
        result = diagnostics.run()
        diagnostics.write_outputs(result)
        recommendation = result.recommendation.to_dicts()[0]
        print("Built ABM v4 essential-input dependence diagnostics.")
        print(f"Recommendation: {recommendation['recommendation']}")
        print(f"Supplier-buyer rows: {result.supplier_buyer_panel.height}")
        print(f"Phase 22 report: {paths.essential_input_dependence_report_path}")
        return

    if args.test_essential_input_dampener:
        if not args.create_output_dirs:
            raise SystemExit(
                "--test-essential-input-dampener requires --create-output-dirs to write validation outputs."
            )
        tester = EssentialInputDampenerTester(paths=paths)
        result = tester.run()
        tester.write_outputs(result)
        recommendation = result.recommendation.to_dicts()[0]
        print("Built ABM v4 essential-input dampener candidate diagnostics.")
        print(f"Recommendation: {recommendation['recommendation']}")
        print(f"Candidates: {result.candidate_grid.height}")
        print(f"Phase 23 report: {paths.essential_input_dampener_report_path}")
        return

    if args.diagnose_eid_failure_modes:
        if not args.create_output_dirs:
            raise SystemExit(
                "--diagnose-eid-failure-modes requires --create-output-dirs to write validation outputs."
            )
        diagnostics = EssentialInputFailureModeDiagnostics(paths=paths)
        result = diagnostics.run()
        diagnostics.write_outputs(result)
        recommendation = result.recommendation.to_dicts()[0]
        print("Built ABM v4 EID dampener failure-mode diagnostics.")
        print(f"Recommendation: {recommendation['recommendation']}")
        print(f"High-EID rows: {result.heterogeneity_panel.height}")
        print(f"Phase 24 report: {paths.eid_failure_mode_report_path}")
        return

    if args.diagnose_adaptive_EID_calibration:
        if not args.create_output_dirs:
            raise SystemExit(
                "--diagnose-adaptive-EID-calibration requires --create-output-dirs to write validation outputs."
            )
        diagnostics = AdaptiveEIDCalibrationDiagnostics(paths=paths)
        result = diagnostics.run()
        diagnostics.write_outputs(result)
        recommendation = result.recommendation.to_dicts()[0]
        print("Built ABM v4 adaptive EID calibration diagnostics.")
        print(f"Recommendation: {recommendation['recommendation']}")
        print(f"Windows: {result.windows.height}")
        print(f"Phase 26 report: {paths.adaptive_EID_report_path}")
        return

    if args.audit_q_energy_mix:
        if not args.create_output_dirs:
            raise SystemExit(
                "--audit-q-energy-mix requires --create-output-dirs to write validation outputs."
            )
        audit = QEnergyMixAudit(paths=paths, start_year=config.start_year, end_year=config.end_year)
        result = audit.run()
        audit.write_outputs(result)
        recommendation = result.recommendation.to_dicts()[0]
        print("Built ABM v4 Q energy-mix audit diagnostics.")
        print(f"Recommendation: {recommendation['recommendation']}")
        print(f"Energy mix rows: {result.energy_mix_panel.height}")
        print(f"Phase 27 report: {paths.q_energy_mix_report_path}")
        return

    if args.finalize_abm_v4:
        if not args.create_output_dirs:
            raise SystemExit(
                "--finalize-abm-v4 requires --create-output-dirs to write validation outputs."
            )
        consolidator = ABMV4FinalConsolidator(paths)
        result = consolidator.run()
        consolidator.write_outputs(result)
        readiness = result.scenario_readiness_assessment.filter(
            pl.col("readiness_dimension") == "overall_scenario_readiness"
        ).to_dicts()[0]
        print("Built ABM v4 final consolidation outputs.")
        print(f"Surviving rules: {result.surviving_rule_comparison.height}")
        print(f"Overall scenario readiness: {readiness['status']}")
        print(f"Final report: {paths.final_abm_v4_consolidation_report_path}")
        return

    if args.build_final_abm_v4_plots_tables:
        if not args.create_output_dirs:
            raise SystemExit(
                "--build-final-abm-v4-plots-tables requires --create-output-dirs to write final artifacts."
            )
        builder = ABMV4FinalArtifactBuilder(paths)
        result = builder.run(write_outputs=True)
        print("Built ABM v4 final plots, clean tables, and artifact index.")
        print(f"Final tables: {len(result.table_paths)}")
        print(f"Final plot files: {len(result.plot_paths)}")
        print(f"Portfolio plot copies: {len(result.copied_plot_paths)}")
        print(f"Artifact index: {result.artifact_index_path}")
        return

    if args.create_output_dirs:
        paths.ensure_output_directories()

    report = inspect_base_model_readiness(paths=paths, config=config)

    print(report.state_source.message)
    print(f"Can run base model: {report.can_run_base_model}")


if __name__ == "__main__":
    main()
