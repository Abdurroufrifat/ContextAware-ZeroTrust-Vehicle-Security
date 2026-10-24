from coact_cps.data.hai23_catalog import (
    HAI23_TEST1_ATTACKS,
    HAI23_TEST2_ATTACKS,
    observable_targets,
)


def test_hai23_catalog_has_official_scenario_counts_and_unique_ids() -> None:
    attacks = HAI23_TEST1_ATTACKS + HAI23_TEST2_ATTACKS
    assert len(HAI23_TEST1_ATTACKS) == 14
    assert len(HAI23_TEST2_ATTACKS) == 38
    assert len(attacks) == 52
    assert len({attack.attack_id for attack in attacks}) == 52


def test_observable_targets_applies_documented_alias() -> None:
    attack = next(item for item in HAI23_TEST2_ATTACKS if item.attack_id == "A213")
    assert observable_targets(attack, {"P1_PP04SP"}) == ("P1_PP04SP",)


def test_combination_attack_preserves_multiple_targets() -> None:
    attack = next(item for item in HAI23_TEST2_ATTACKS if item.attack_id == "A228")
    assert attack.primitives == ("AP05", "AP30")
    assert observable_targets(attack, {"P1_PCV01D", "P2_SCO"}) == (
        "P1_PCV01D",
        "P2_SCO",
    )
