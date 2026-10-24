from pathlib import Path
import importlib.util,copy,pytest
def core():
 p=Path(__file__).with_name("input_contract.py");assert p.exists(),"neural feature admission missing"
 s=importlib.util.spec_from_file_location("input_contract",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def field(quantity,unit,frame="sensor"):
 return {"quantity":quantity,"unit":unit,"frame":frame,"semantics_verified":True,"evidence":"reviewed source mapping"}
def available():
 return {"x":field("position","m"),"y":field("position","m"),"z":field("position","m"),"doppler":field("raw_radial_velocity","m/s"),"rcs":{"quantity":"radar_cross_section","unit":None,"frame":None,"semantics_verified":False,"evidence":"unsigned byte only"}}
def test_missing_power_blocks_msc3d():
 r=core().check("msc3d",available());assert not r["admitted"] and any(x["field"]=="power" for x in r["reasons"])
def test_missing_snr_blocks_nav3d():
 r=core().check("nav3d",available());assert not r["admitted"] and any(x["field"]=="snr" for x in r["reasons"])
def test_unknown_rcs_units_block_radarscenes():
 r=core().check("radarscenes2d",available());assert not r["admitted"] and any(x["field"]=="rcs" for x in r["reasons"])
def test_a_genuinely_documented_snr_channel_can_pass():
 a=available();a["snr"]=field("signal_to_noise_ratio","dB",None);assert core().check("nav3d",a)["admitted"]
def test_rcs_cannot_be_renamed_to_snr():
 a=available();a["snr"]=field("radar_cross_section","dB",None);assert not core().check("nav3d",a)["admitted"]
def test_correct_rcs_semantics_can_admit_2d():
 a=available();a["rcs"]=field("radar_cross_section","dBsm",None);assert core().check("radarscenes2d",a)["admitted"]
def test_same_numeric_shape_with_vehicle_coordinates_is_not_sensor_frame():
 a=available();a["snr"]=field("signal_to_noise_ratio","dB",None);a["x"]["frame"]="vehicle";assert not core().check("nav3d",a)["admitted"]
def test_label_cannot_masquerade_as_radial_measurement():
 a=available();a["snr"]=field("signal_to_noise_ratio","dB",None);a["doppler"]["quantity"]="reference_speed";assert not core().check("nav3d",a)["admitted"]
def test_declared_units_without_verified_mapping_still_block():
 a=available();a["rcs"]=field("radar_cross_section","dBsm",None);a["rcs"]["semantics_verified"]=False;assert not core().check("radarscenes2d",a)["admitted"]
def test_verified_flag_without_evidence_does_not_pass():
 a=available();a["rcs"]=field("radar_cross_section","dBsm",None);a["rcs"]["evidence"]="";assert not core().check("radarscenes2d",a)["admitted"]
def test_unknown_model_name_not_silently_accepted():
 with pytest.raises(ValueError):core().check("unknown",available())
