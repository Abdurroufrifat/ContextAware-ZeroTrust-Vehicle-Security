"""Strict recorded-reference parsing and coverage reporting; no estimator access."""
import csv,io
import numpy as np
FIELDS="GPSTime,easting,northing,altitude,vel_east,vel_north,vel_up,roll,pitch,heading,angvel_z,angvel_y,angvel_x".split(",")
def parse_reference(raw):
    try:
        records=list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
        if not records or records[0]!=FIELDS or len(records)<3:raise ValueError("Reference requires header and at least two rows")
        if any(len(row)!=13 for row in records[1:]):raise ValueError("Reference column count")
        times=np.array([int(row[0]) for row in records[1:]],dtype=np.int64)
        velocities=np.array([[float(x) for x in row[4:7]] for row in records[1:]],dtype=np.float64)
        if not np.all(np.diff(times)>0) or not np.isfinite(velocities).all():raise ValueError("Invalid reference times or velocities")
        return times,velocities
    except (UnicodeError,TypeError,OverflowError) as exc:
        raise ValueError("Malformed reference") from exc
def comparison_completed(rows,expected_rows):
    return bool(expected_rows>0 and len(rows)==expected_rows and
                all(row["reference_supported"] is True for row in rows) and
                any(row["scored"] is True for row in rows))
