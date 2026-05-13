from src.abm_v4.ecosystem import EcosystemAssignment


def test_ecosystem_assignment_validates_source_vocabulary() -> None:
    assignment = EcosystemAssignment(
        sector="Agriculture",
        ecosystem_id="eco_food",
        ecosystem_label="Food system",
        ecosystem_source="eora_sector_manual_mapping",
    )

    assert assignment.is_known_source()
