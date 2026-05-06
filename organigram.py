from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RepoNode:
    label: str
    path: str | None = None
    children: tuple["RepoNode", ...] = field(default_factory=tuple)

    def display_label(self) -> str:
        if self.path:
            return f"{self.label}\\n{self.path}"
        return self.label


class OrganigramBuilder:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def exists(self, relative_path: str) -> bool:
        return (self.repo_root / relative_path).exists()

    def keep_existing_nodes(self, nodes: Iterable[RepoNode]) -> tuple[RepoNode, ...]:
        kept_nodes: list[RepoNode] = []

        for node in nodes:
            children = self.keep_existing_nodes(node.children)

            if node.path is None or self.exists(node.path) or children:
                kept_nodes.append(
                    RepoNode(
                        label=node.label,
                        path=node.path,
                        children=children,
                    )
                )

        return tuple(kept_nodes)

    def build_architecture(self) -> RepoNode:
        architecture = RepoNode(
            label="Remaster repository architecture",
            path=".",
            children=(
                RepoNode(
                    label="Raw Eora26 MRIO data",
                    path="data/parquet",
                    children=(
                        RepoNode("Intermediate transactions", "data/parquet/{year}/T.parquet"),
                        RepoNode("Final demand", "data/parquet/{year}/FD.parquet"),
                        RepoNode("Environmental extensions", "data/parquet/{year}/Q.parquet"),
                        RepoNode("Value added", "data/parquet/{year}/VA.parquet"),
                    ),
                ),
                RepoNode(
                    label="Raw Eora labels",
                    path="data/raw",
                    children=(
                        RepoNode("Transaction labels", "data/raw/{year}/labels_T.parquet"),
                        RepoNode("Final-demand labels", "data/raw/{year}/labels_FD.parquet"),
                        RepoNode("Environmental labels", "data/raw/{year}/labels_Q.parquet"),
                    ),
                ),
                RepoNode(
                    label="Atlas of Economic Complexity bridge",
                    path="data/atlas",
                    children=(
                        RepoNode("Raw country-product-year data", "data/atlas/raw/country_product_year"),
                        RepoNode("HS92 to Eora26 concordance", "data/atlas/concordance/hs92_to_eora26_prefilled.csv"),
                        RepoNode("Clean HS92 Atlas panel", "data/atlas/processed/atlas_hs92_level4_clean_panel_1995_2016.parquet"),
                        RepoNode("Eora26 sector capability panel", "data/atlas/processed/atlas_eora26_sector_capabilities_1995_2016.parquet"),
                        RepoNode("Country-sector labels", "data/atlas/processed/eora26_country_sector_labels.csv"),
                    ),
                ),
                RepoNode(
                    label="Yearly Eora-derived metrics",
                    path="data/metrics",
                    children=(
                        RepoNode("Emissions intensity", "data/metrics/{year}/ei_{year}.parquet"),
                        RepoNode("Embodied emissions transfer matrix", "data/metrics/{year}/et_{year}.parquet"),
                        RepoNode("Network centrality metrics", "data/metrics/{year}/centrality_{year}.parquet"),
                        RepoNode("Network green-ness metrics", "data/metrics/{year}/greenness_{year}.parquet"),
                    ),
                ),
                RepoNode(
                    label="ABM data layer",
                    path="data/abm",
                    children=(
                        RepoNode("ABM input panel", "data/abm/metrics/abm_metrics_panel.parquet"),
                        RepoNode("Transition diagnostics", "data/abm/diagnostics"),
                        RepoNode("Clean transition model outputs", "data/abm/model_outputs_clean"),
                        RepoNode("Scenario simulation outputs", "data/abm/scenarios"),
                    ),
                ),
                RepoNode(
                    label="Source code",
                    path="src",
                    children=(
                        RepoNode(
                            label="Metric builder",
                            path="src/metric_builder",
                            children=(
                                RepoNode("Computes EI, ET, centrality, green-ness", "src/metric_builder/compute_metrics.py"),
                            ),
                        ),
                        RepoNode(
                            label="Atlas / modelling bridge",
                            path="src/modelling",
                            children=(
                                RepoNode("Merge Eora and Atlas capability data", "src/modelling/merge_eora_atlas.py"),
                            ),
                        ),
                        RepoNode(
                            label="Agent-based model",
                            path="src/abm",
                            children=(
                                RepoNode("Input assembly", "src/abm/metrics.py"),
                                RepoNode("State update rules", "src/abm/dynamics.py"),
                                RepoNode("Scenario definitions", "src/abm/scenarii.py"),
                                RepoNode("ABM class", "src/abm/model.py"),
                                RepoNode("Simulation runner", "src/abm/runner.py"),
                                RepoNode("ABM plots", "src/abm/plots.py"),
                            ),
                        ),
                        RepoNode(
                            label="Plotting utilities",
                            path="src/plotting",
                        ),
                    ),
                ),
                RepoNode(
                    label="Outputs",
                    path="outputs",
                    children=(
                        RepoNode("Generated figures", "outputs/plots"),
                    ),
                ),
                RepoNode(
                    label="Documentation",
                    path="docs",
                    children=(
                        RepoNode("Conceptual notes"),
                        RepoNode("LaTeX drafts"),
                        RepoNode("Methodological writeups"),
                    ),
                ),
            ),
        )

        return RepoNode(
            label=architecture.label,
            path=architecture.path,
            children=self.keep_existing_nodes(architecture.children),
        )


class MermaidRenderer:
    def __init__(self) -> None:
        self.lines: list[str] = ["flowchart TD"]
        self.counter = 0

    def next_id(self) -> str:
        self.counter += 1
        return f"N{self.counter}"

    def sanitize_label(self, label: str) -> str:
        return (
            label.replace('"', "'")
            .replace("{", "&#123;")
            .replace("}", "&#125;")
        )

    def render(self, root: RepoNode) -> str:
        self.lines = ["flowchart TD"]
        self.counter = 0
        self._render_node(root)
        return "\n".join(self.lines)

    def _render_node(self, node: RepoNode, parent_id: str | None = None) -> str:
        node_id = self.next_id()
        label = self.sanitize_label(node.display_label())

        self.lines.append(f'    {node_id}["{label}"]')

        if parent_id is not None:
            self.lines.append(f"    {parent_id} --> {node_id}")

        for child in node.children:
            self._render_node(child, node_id)

        return node_id


class MarkdownOrganigramWriter:
    def __init__(self, builder: OrganigramBuilder, renderer: MermaidRenderer) -> None:
        self.builder = builder
        self.renderer = renderer

    def write(self, output_path: Path) -> None:
        architecture = self.builder.build_architecture()
        mermaid = self.renderer.render(architecture)

        content = (
            "# Remaster Repository Organigram\n\n"
            "This organigram focuses on the parts of the repository used in the active modelling workflow.\n\n"
            "```mermaid\n"
            f"{mermaid}\n"
            "```\n"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_path = repo_root / "docs" / "repo_organigram.md"

    builder = OrganigramBuilder(repo_root=repo_root)
    renderer = MermaidRenderer()
    writer = MarkdownOrganigramWriter(builder=builder, renderer=renderer)

    writer.write(output_path)

    print(f"Organigram written to: {output_path}")


if __name__ == "__main__":
    main()