from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ABMV5Paths:
    """Centralized ABM v5 paths without implicit directory creation."""

    project_root: Path

    @classmethod
    def from_project_root(cls, project_root: Path | str) -> "ABMV5Paths":
        """Create an ABM v5 path registry from a project root path."""
        return cls(project_root=Path(project_root))

    @property
    def data_abm_v5(self) -> Path:
        return self.project_root / "data" / "abm_v5"

    @property
    def inputs(self) -> Path:
        return self.data_abm_v5 / "inputs"

    @property
    def accounting(self) -> Path:
        return self.data_abm_v5 / "accounting"

    @property
    def phase_space(self) -> Path:
        return self.data_abm_v5 / "phase_space"

    @property
    def regimes(self) -> Path:
        return self.data_abm_v5 / "regimes"

    @property
    def capabilities(self) -> Path:
        return self.data_abm_v5 / "capabilities"

    @property
    def supplier_network(self) -> Path:
        return self.data_abm_v5 / "supplier_network"

    @property
    def energy_inertia(self) -> Path:
        return self.data_abm_v5 / "energy_inertia"

    @property
    def policy(self) -> Path:
        return self.data_abm_v5 / "policy"

    @property
    def simulation(self) -> Path:
        return self.data_abm_v5 / "simulation"

    @property
    def validation(self) -> Path:
        return self.data_abm_v5 / "validation"

    @property
    def diagnostics(self) -> Path:
        return self.data_abm_v5 / "diagnostics"

    @property
    def reports(self) -> Path:
        return self.data_abm_v5 / "reports"

    @property
    def plots(self) -> Path:
        return self.project_root / "outputs" / "plots" / "abm_v5"

    @property
    def plots_diagnostics(self) -> Path:
        return self.plots / "diagnostics"

    @property
    def plots_research(self) -> Path:
        return self.plots / "research"

    @property
    def plots_portfolio(self) -> Path:
        return self.plots / "portfolio"

    @property
    def logs(self) -> Path:
        return self.project_root / "logs" / "abm_v5"

    @property
    def tmp(self) -> Path:
        return self.project_root / "tmp" / "abm_v5"

    @property
    def docs(self) -> Path:
        return self.project_root / "docs" / "abm_v5"

    def as_dict(self) -> dict[str, Path]:
        """Return all named ABM v5 paths."""
        return {
            "project_root": self.project_root,
            "data_abm_v5": self.data_abm_v5,
            "inputs": self.inputs,
            "accounting": self.accounting,
            "phase_space": self.phase_space,
            "regimes": self.regimes,
            "capabilities": self.capabilities,
            "supplier_network": self.supplier_network,
            "energy_inertia": self.energy_inertia,
            "policy": self.policy,
            "simulation": self.simulation,
            "validation": self.validation,
            "diagnostics": self.diagnostics,
            "reports": self.reports,
            "plots": self.plots,
            "plots_diagnostics": self.plots_diagnostics,
            "plots_research": self.plots_research,
            "plots_portfolio": self.plots_portfolio,
            "logs": self.logs,
            "tmp": self.tmp,
            "docs": self.docs,
        }

    def validate_project_root(self) -> None:
        """Raise if the project root does not look like this repository."""
        pyproject_path = self.project_root / "pyproject.toml"
        if not pyproject_path.is_file():
            raise FileNotFoundError(
                f"Expected pyproject.toml under project_root: {self.project_root}"
            )

    def ensure_directories(self) -> None:
        """Create all expected ABM v5 directories."""
        for path_name, directory_path in self.as_dict().items():
            if path_name == "project_root":
                continue
            directory_path.mkdir(parents=True, exist_ok=True)
