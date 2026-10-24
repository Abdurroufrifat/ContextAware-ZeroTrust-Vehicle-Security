"""Evaluation labels must come from generated truth, never from MAC success."""
import gzip,json
from pathlib import Path
def test_anchor_error_premise_is_numerically_true():
 root=Path(__file__).resolve().parent
 with gzip.open(root/"decisions.jsonl.gz","rt",encoding="utf-8") as f:
  for line in f:
   row=json.loads(line);packet=row["packets"]["anchor"]
   actual=False if packet is None else abs(json.loads(packet["wire"])["body"]["speed_mmps"]-10000)<=250
   assert row["evaluation"]["anchor_bound_holds"] is actual,(row["condition"],row["step"],row["case"])
