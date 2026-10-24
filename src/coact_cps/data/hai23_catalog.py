from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HAI23Attack:
    attack_id: str
    primitives: tuple[str, ...]
    target_controllers: tuple[str, ...]
    target_points: tuple[str, ...]


def _attack(
    attack_id: str,
    primitives: str,
    controllers: str,
    points: str,
) -> HAI23Attack:
    return HAI23Attack(
        attack_id=attack_id,
        primitives=tuple(primitives.split(";")),
        target_controllers=tuple(controllers.split(";")),
        target_points=tuple(points.split(";")),
    )


# Transcribed from pages 31-32 of the official HAI Dataset Technical Details
# manual (v4.0, May 31, 2023). Ordering matches the finalized label-test1.csv
# and label-test2.csv attack episodes.
HAI23_TEST1_ATTACKS = (
    _attack("A101", "AP01", "P1-PC-SP1", "P1_B2016"),
    _attack("A102", "AP02", "P1-PC-SP1PV1", "P1_B2016;P1_PIT01"),
    _attack("A103", "AP04", "P1-PC-CO1", "P1_PCV01D"),
    _attack("A104", "AP05", "P1-PC-CO1PV1", "P1_PCV01D;P1_PIT01"),
    _attack("A105", "AP07", "P1-PC-CO1-ST", "P1_PCV01D"),
    _attack("A106", "AP03", "P1-PC-SP1PV1PV2", "P1_B2016;P1_PIT01"),
    _attack("A107", "AP40", "P1-PC-SP1-LT", "P1_B2016"),
    _attack("A108", "AP08", "P1-FC-SP1", "P1_B3005"),
    _attack("A109", "AP09", "P1-FC-SP1PV1", "P1_B3005;P1_FT03"),
    _attack("A110", "AP11", "P1-FC-CO1", "P1_FCV03D"),
    _attack("A111", "AP12", "P1-FC-CO1PV1", "P1_FCV03D;P1_FT03"),
    _attack("A112", "AP13", "P1-FC-CO1-ST", "P1_FCV03D"),
    _attack("A113", "AP10", "P1-FC-SP1PV1PV2", "P1_B3005;P1_FT03;P1_LIT01"),
    _attack("A114", "AP41", "P1-FC-SP1-LT", "P1_B3005"),
)


HAI23_TEST2_ATTACKS = (
    _attack("A201", "AP14", "P1-LC-SP1", "P1_B3004"),
    _attack("A202", "AP15", "P1-LC-SP1PV1", "P1_B3004;P1_LIT01"),
    _attack("A203", "AP16", "P1-LC-CO1", "P1_LCV01D"),
    _attack("A204", "AP17", "P1-LC-CO1PV1", "P1_LCV01D;P1_LIT01"),
    _attack("A205", "AP18", "P1-LC-CO1-ST", "P1_LCV01D"),
    _attack("A206", "AP03", "P1-PC-SP1PV1PV2", "P1_B2016;P1_PIT01;P1_FIT"),
    _attack("A207", "AP43", "P1-LC-CO1-LT", "P1_LCV01D"),
    _attack("A208", "AP42", "P1-LC-CO1PV1PV2", "P1_LCV01D;P1_LIT01;P1_FIT"),
    _attack("A209", "AP23", "P1-CC-CO1", "P1_PP04"),
    _attack("A210", "AP23", "P1-CC-CO1", "P1_PP04"),
    _attack("A211", "AP46", "P1-CC-CO1PV1", "P1_PP04;P1_TIT03"),
    _attack("A212", "AP24", "P1-CC-CO1-ST", "P1_PP04"),
    _attack("A213", "AP25", "P1-CC-SP1-LT", "P1_PP04_SP"),
    _attack("A214", "AP19", "P1-TC-CO1", "P1_FCV01D"),
    _attack("A215", "AP20", "P1-TC-CO1PV1", "P1_FCV01D;P1_TIT01"),
    _attack("A216", "AP21", "P1-TC-CO1-ST", "P1_FCV01D"),
    _attack("A217", "AP22", "P1-TC-SP1-LT", "P1_B4002"),
    _attack("A218", "AP07", "P1-PC-CO1-ST", "P1_PCV01D"),
    _attack("A219", "AP13", "P1-FC-CO1-ST", "P1_FCV03D"),
    _attack("A220", "AE03", "P1-LC-CO-AVALGEN", "1002-14-AVALGEN.R1"),
    _attack("A221", "AE08", "P1-HC-SP-LOWMON", "1004-21-LOWMON.R1"),
    _attack("A222", "AE01", "P1-PC-CO-AVALGEN", "1001-09-AVALGEN.R1"),
    _attack("A223", "AE07", "P1-CC-CO-FUNCTION", "1020-15-FUNCTION.S4"),
    _attack("A224", "AP14;AP26", "P1-LC-SP1;P2-SC-SP1", "P1_B3004;P2_AutoSD"),
    _attack("A225", "AP16;AP32", "P1-LC-CO1;P2-TC-SP1", "P1_LCV01D;P2_VTR01"),
    _attack("A226", "AP04;AP11", "P1-PC-CO1;P1-FC-CO1", "P1_PCV01D;P1_FCV03D"),
    _attack(
        "A227",
        "AP09;AP14",
        "P1-FC-SP1PV1;P1-LC-SP1",
        "P1_B3005;P1_FT03;P1_B3004",
    ),
    _attack(
        "A228",
        "AP05;AP30",
        "P1-PC-CO1PV1;P2-SC-CO1PV1",
        "P1_PCV01D;P1_PIT01;P2_SCO;P2_SIT01",
    ),
    _attack("A229", "AP45;AP01", "P1-TC-SP1;P1-PC-SP1", "P1_B4002;P1_B2016"),
    _attack(
        "A230",
        "AP19;AP02",
        "P1-TC-CO1;P1-PC-SP1PV1",
        "P1_FCV01D;P1_B2016;P1_PIT01",
    ),
    _attack("A231", "AP08;AP35", "P1-FC-SP1;P3-LC-CO1", "P1_B3005;P3_LCP01D"),
    _attack(
        "A232",
        "AP45;AP27",
        "P1-TC-SP1;P2-SC-SP1PV1",
        "P1_B4002;P2_AutoSD;P2_SIT01",
    ),
    _attack(
        "A233",
        "AP44;AP47",
        "P1-LC-CO1PV1-LT;P2-TC-SP2-LT",
        "P1_LCV01D;P1_LIT01;P2_VTR02",
    ),
    _attack("A234", "AP25", "P1-CC-SP1-LT", "P1_PP04_SP"),
    _attack("A235", "AE05", "P1-TC-PV_FUNCTION1", "1003-05-FUNCTION.T9"),
    _attack("A236", "AE06", "P1-TC-PV_FUNCTION2", "OCB0004018.T6"),
    _attack("A237", "AE05", "P1-PC-CO_FUNCTION", "OCB0002006.S5"),
    _attack("A238", "AE06", "P1-LC-CO_FUNCTION", "1002-31-FUNCTION.S5"),
)


HAI23_ATTACKS_BY_LABEL_FILE = {
    "label-test1.csv": HAI23_TEST1_ATTACKS,
    "label-test2.csv": HAI23_TEST2_ATTACKS,
}


TARGET_ALIASES = {
    "P1_PP04_SP": "P1_PP04SP",
}


def observable_targets(attack: HAI23Attack, feature_names: set[str]) -> tuple[str, ...]:
    return tuple(
        alias
        for point in attack.target_points
        if (alias := TARGET_ALIASES.get(point, point)) in feature_names
    )
