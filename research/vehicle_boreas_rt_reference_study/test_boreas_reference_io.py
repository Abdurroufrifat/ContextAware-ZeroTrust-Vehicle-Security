import pytest
def api():
 import boreas_reference_io as m
 return m
HEADER="GPSTime,easting,northing,altitude,vel_east,vel_north,vel_up,roll,pitch,heading,angvel_z,angvel_y,angvel_x\n"
def test_singleton_reference_is_rejected_as_schema_error():
 with pytest.raises(ValueError):api().parse_reference((HEADER+"1,0,0,0,3,4,0,0,0,0,0,0,0\n").encode())
def test_header_only_reference_is_rejected_as_schema_error():
 with pytest.raises(ValueError):api().parse_reference(HEADER.encode())
def test_integer_epoch_times_preserved():
 raw=(HEADER+"1733249737399406,0,0,0,3,4,0,0,0,0,0,0,0\n1733249737499406,0,0,0,0,0,5,0,0,0,0,0,0\n").encode()
 t,v=api().parse_reference(raw);assert t.tolist()==[1733249737399406,1733249737499406] and v.tolist()==[[3,4,0],[0,0,5]]
@pytest.mark.parametrize("rows,complete",[
 ([],False),
 ([{"reference_supported":True,"scored":True}],False),
 ([{"reference_supported":True,"scored":True},{"reference_supported":False,"scored":False}],False),
 ([{"reference_supported":True,"scored":False},{"reference_supported":True,"scored":False}],False),
 ([{"reference_supported":True,"scored":True},{"reference_supported":True,"scored":False}],True)])
def test_completion_requires_expected_inventory_reference_support_and_actual_score(rows,complete):
 assert api().comparison_completed(rows,2) is complete
