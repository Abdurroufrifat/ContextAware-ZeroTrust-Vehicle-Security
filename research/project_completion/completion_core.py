"""Completion is separate from passing software tests; unknown evidence never passes."""
import math
def complete(requirements):
 return bool(requirements) and all(status=="passed" for status in requirements.values())
def metric_gate(accepted,requests,p95,large_errors):
 return bool(requests>0 and 0<accepted<=requests and accepted/requests>=.95 and p95 is not None
  and math.isfinite(p95) and 0<=p95<=.5 and large_errors==0)
