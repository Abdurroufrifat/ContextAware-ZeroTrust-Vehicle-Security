from common import *
install_guard()
import csv,collections
r=json.loads((R/"results.json").read_text());v=json.loads((R/"verification.json").read_text())
assert v["passed"]
groups=collections.defaultdict(list)
for cell in r["conditions"]:groups[cell["case"]].append(cell)
summary=[]
for case,cells in groups.items():
 summary.append({"case":case,"queries":sum(c["queries"] for c in cells),"selected":sum(c["selected"] for c in cells),
 "error_above500":sum(c["error_above500"] for c in cells),"error_above2000":sum(c["error_above2000"] for c in cells),
 "conditional_error_above500":sum(c["conditional_error_above500"] for c in cells),"replayed_evidence_outputs":sum(c["replayed_evidence_outputs"] for c in cells)})
with (R/"case_summary.csv").open("w",newline="",encoding="utf-8") as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
with (R/"all_conditions.csv").open("w",newline="",encoding="utf-8") as f:
 rows=[{k:v for k,v in c.items() if k!="selections"} for c in r["conditions"]]
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
totals={k:sum(c[k] for c in summary) for k in ("queries","selected","error_above500","error_above2000","conditional_error_above500","replayed_evidence_outputs")}
write("summary.json",{"totals":totals,"cases":summary,"physical_validation_established":False,"specific_novelty_established":False,"promotion_allowed":False})
print(json.dumps({"totals":totals,"negative_cases":[c for c in summary if c["error_above500"] or c["replayed_evidence_outputs"]]},indent=2))
