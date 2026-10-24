from ambiguity_guard import corroborate
def b():return {"conditional_budget_satisfied":True,"low_n":14872,"low_d":1,"high_n":15497,"high_d":1,"research_speed_mmps":15121}
def test_scene_spoofing_conflict_is_not_promoted():
 assert corroborate(21088,b())==(None,"unresolved_source_conflict")
def test_disturbed_base_does_not_overwrite_honest_scene():
 d=b();d.update(low_n=17872,high_n=18497,research_speed_mmps=18121)
 assert corroborate(15090,d)==(None,"unresolved_source_conflict")
def test_compatible_learning_is_retained():
 assert corroborate(15090,b())==(15090,"corroborated")
