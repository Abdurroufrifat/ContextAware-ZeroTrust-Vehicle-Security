"""Causal descriptive comparison; no control or hard-error certificate."""
from bisect import bisect_right
from math import isfinite, hypot

def pair_samples(commands, references, max_age_ns):
    if type(max_age_ns) is not int or max_age_ns < 0:
        raise ValueError("age must be a nonnegative integer")
    for stream, strict, width in [(commands, True, 2), (references, False, 4)]:
        last = None
        for row in stream:
            if len(row) != width or type(row[0]) is not int:
                raise ValueError("invalid row or timestamp")
            if last is not None and (row[0] < last or (strict and row[0] == last)):
                raise ValueError("timestamps are not ordered uniquely for commands")
            if any(isinstance(x, bool) or not isfinite(x) for x in row[1:]):
                raise ValueError("nonfinite measurement")
            last = row[0]
    times = [r[0] for r in commands]
    out = []
    for k, (t, x, y, z) in enumerate(references):
        j = bisect_right(times, t) - 1
        row = {"reference_index": k, "reference_time_ns": t,
               "reference_mps": hypot(x, y, z), "command_index": j}
        if j < 0:
            row.update(status="unmatched", command_time_ns=None, command_signed_mps=None,
                       command_mps=None, age_ns=None, error_mps=None)
        else:
            ct, value = commands[j]
            age = t - ct
            row.update(status="fresh" if age <= max_age_ns else "stale",
                       command_time_ns=ct, command_signed_mps=value, command_mps=abs(value),
                       age_ns=age, error_mps=abs(abs(value)-row["reference_mps"]))
        out.append(row)
    return out


def longitudinal_commands(records):
    if any(r["linear_y"]!=0 or r["linear_z"]!=0 for r in records):
        raise ValueError("nonlongitudinal commands outside assay scope")
    return [(r["time_ns"],r["linear_x"]) for r in records]
