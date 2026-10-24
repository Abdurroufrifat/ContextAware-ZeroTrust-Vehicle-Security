"""Time-aware saved-decision metrics; observed spans are not token lifetimes."""

import numpy as np


def runs(mask, start, time):
    result, begin = [], None
    for i, selected in enumerate(mask):
        contiguous = i > 0 and start[i] == time[i - 1]
        if begin is not None and (not selected or not contiguous):
            result.append((begin, i))
            begin = None
        if selected and begin is None:
            begin = i
    if begin is not None:
        result.append((begin, len(mask)))
    return result


def summarize(time, start, labels, positions, alarm, base, grant):
    time, start, labels, positions, alarm, base, grant = map(
        np.asarray, (time, start, labels, positions, alarm, base, grant)
    )
    n = len(time)
    if (
        n == 0
        or time.shape != (n,)
        or start.shape != (n,)
        or time.dtype.kind not in "iu"
        or start.dtype.kind not in "iu"
        or any(x.shape != (n,) or x.dtype.kind != "b" for x in (labels, base, grant))
        or positions.ndim != 1
        or positions.dtype.kind not in "iu"
        or alarm.shape != positions.shape
        or alarm.dtype.kind != "b"
        or np.any(positions >= n)
        or np.any(positions < 0)
        or np.any(np.diff(positions.astype(np.int64)) <= 0)
        or np.any(time <= start)
        or np.any(np.diff(time.astype(np.int64)) <= 0)
        or np.any(start[1:] < time[:-1])
    ):
        raise ValueError("aligned boolean decisions and ordered non-overlapping intervals required")
    expected = np.zeros(n, dtype=bool)
    expected[positions] = base[positions] & ~alarm
    if not np.array_equal(grant, expected):
        raise ValueError("saved grant does not match scored base policy and alarms")
    neural, scored = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    neural[positions], scored[positions] = alarm, True
    duration = time.astype(np.int64) - start.astype(np.int64)
    normal, active_grant = ~labels, labels & grant
    normal_bursts = runs(normal & neural, start, time)
    normal_denials = runs(normal & ~grant, start, time)
    grant_runs = runs(active_grant, start, time)
    events = []
    for begin, end in runs(labels, start, time):
        hits = np.flatnonzero(neural[begin:end]) + begin
        denials = np.flatnonzero(~grant[begin:end]) + begin
        first_alarm = int(hits[0]) if len(hits) else None
        first_denial = int(denials[0]) if len(denials) else None
        events.append(
            dict(
                first_epoch=begin,
                last_epoch=end - 1,
                start_ms=int(start[begin]),
                end_ms=int(time[end - 1]),
                epochs=end - begin,
                scored_epochs=int(scored[begin:end].sum()),
                first_neural_alarm_delay_ms=(
                    int(time[first_alarm]) - int(start[begin]) if first_alarm is not None else None
                ),
                first_denial_delay_ms=(
                    int(time[first_denial]) - int(start[begin])
                    if first_denial is not None
                    else None
                ),
                granted_epochs=int(grant[begin:end].sum()),
                granted_epochs_after_first_denial=(
                    int(grant[first_denial + 1 : end].sum()) if first_denial is not None else 0
                ),
                fully_denied=not bool(grant[begin:end].any()),
            )
        )
    return dict(
        epochs=n,
        scored_epochs=len(positions),
        unscored_epochs=n - len(positions),
        normal_epochs=int(normal.sum()),
        normal_scored_epochs=int((normal & scored).sum()),
        normal_granted_epochs=int((normal & grant).sum()),
        normal_denied_scored_epochs=int((normal & scored & ~grant).sum()),
        normal_neural_alarm_bursts=len(normal_bursts),
        normal_neural_longest_alarm_run_epochs=max((b - a for a, b in normal_bursts), default=0),
        normal_longest_denial_run_epochs=max((b - a for a, b in normal_denials), default=0),
        normal_granted_observed_span_ms=int(duration[normal & grant].sum()),
        attack_epochs=int(labels.sum()),
        attack_events=len(events),
        neural_missed_events=sum(e["first_neural_alarm_delay_ms"] is None for e in events),
        fully_denied_events=sum(e["fully_denied"] for e in events),
        events_with_grants_after_first_denial=sum(
            e["granted_epochs_after_first_denial"] > 0 for e in events
        ),
        attack_granted_epochs=int(active_grant.sum()),
        attack_granted_observed_span_ms=int(duration[active_grant].sum()),
        attack_longest_grant_run_epochs=max((b - a for a, b in grant_runs), default=0),
        attack_longest_grant_run_observed_span_ms=max(
            (int(duration[a:b].sum()) for a, b in grant_runs), default=0
        ),
        events=events,
    )
