"""Research benchmark admission from reviewed metadata, not source authentication."""
BASE={"x":("position","m","sensor"),"y":("position","m","sensor"),"doppler":("raw_radial_velocity","m/s","sensor")}
REQUIRED={"msc3d":{**BASE,"z":("position","m","sensor"),"power":("native_msc_power",None,None)},"nav3d":{**BASE,"z":("position","m","sensor"),"snr":("signal_to_noise_ratio",None,None)},"radarscenes2d":{**BASE,"rcs":("radar_cross_section","dBsm",None)}}
def check(model,fields):
 if model not in REQUIRED:raise ValueError("unknown native model contract")
 reasons=[]
 for name,(quantity,unit,frame) in REQUIRED[model].items():
  actual=fields.get(name)
  if actual is None:reasons.append({"field":name,"reason":"missing native feature"});continue
  if actual.get("quantity")!=quantity:reasons.append({"field":name,"reason":"different measured quantity"})
  if actual.get("unit") is None or (unit is not None and actual.get("unit")!=unit):reasons.append({"field":name,"reason":"unknown or different units"})
  if frame is not None and actual.get("frame")!=frame:reasons.append({"field":name,"reason":"different coordinate frame"})
  if actual.get("semantics_verified") is not True or not actual.get("evidence"):reasons.append({"field":name,"reason":"native feature mapping not verified"})
 return {"model":model,"admitted":not reasons,"reasons":reasons,"scope":"reviewed cross-dataset feature compatibility only"}
