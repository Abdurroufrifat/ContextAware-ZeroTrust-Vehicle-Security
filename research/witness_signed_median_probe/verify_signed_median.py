"""Offline audit of the single exploratory candidate and immutable predecessor."""
from pathlib import Path
import contextlib,csv,hashlib,io,json,statistics,unittest
from base_loader import BASE,baseline
import verify_probe
import run_candidate
import test_candidate
HERE=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def audit():
    manifest=json.loads((HERE/'sha256.json').read_text())
    for name,expected in manifest.items():
        p=(HERE/name).resolve()
        if not p.is_relative_to(HERE.resolve()) or digest(p)!=expected:raise ValueError('update digest mismatch: '+name)
    inputs=json.loads((HERE/'input_lock.json').read_text())
    for name,expected in inputs['files'].items():
        p=(BASE/name).resolve()
        if not p.is_relative_to(BASE.resolve()) or digest(p)!=expected:raise ValueError('previous calibration package changed: '+name)
    buffer=io.StringIO()
    with contextlib.redirect_stdout(buffer):verify_probe.main()
    expected=json.loads((HERE/'result.json').read_text());actual,rows=run_candidate.compute()
    if actual!=expected:raise ValueError('comparison result changed')
    text=io.StringIO(newline='');writer=csv.DictWriter(text,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    if text.getvalue().encode()!=(HERE/'comparison.csv').read_bytes():raise ValueError('comparison CSV changed')
    radar=run_candidate.read_stream('radar_raw.csv')
    by_window={}
    for r in radar:by_window.setdefault(r['window_id'],[]).append(r)
    for row in rows:
        t=row['time_ns']
        source=[r['speed_raw'] for r in by_window[row['window_id']] if t-1000000000<=r['bag_time_ns']<=t and baseline.valid_sample(r,250000000)]
        if len(source)<10 or abs(statistics.median(source))!=row['feature_raw']:raise ValueError('independent signed-feature reconstruction failed')
    # Recheck available signal quality directly in serialized source messages.
    n=0
    for p in sorted((BASE/'evidence/chunks').glob('radars.bag.*.bin')):
        body=list(baseline.records(p.read_bytes()))[0][1]
        for h,b in baseline.records(body):
            if h['op']!=b'\x02' or baseline.u32(h['conn'])!=4:continue
            cur=baseline.Cursor(b);cur.unpack('<f');cur.string();cur.unpack('<f');cur.string()
            if cur.unpack('<f')[0]!=1:raise ValueError('object count quality finding changed')
            cur.header()
            for _ in range(4):
                length=cur.unpack('<I')[0]
                if length!=0:raise ValueError('raw signal quality finding changed')
            if cur.unpack('<f')[0]!=0 or cur.p!=len(b):raise ValueError('range or schema finding changed')
            n+=1
    if n!=1629:raise ValueError('raw radar cohort changed')
    tests=unittest.TextTestRunner(stream=buffer).run(unittest.defaultTestLoader.loadTestsFromModule(test_candidate))
    if not tests.wasSuccessful():raise ValueError(buffer.getvalue())
    if expected['witness_enabled'] or expected['independent_validation'] or expected['production_policy_changed']:raise ValueError('exploratory scope was exceeded')
    print(f'COMPARISON AUDIT PASS: {len(rows)} unchanged endpoints, {n} raw radar records, {tests.testsRun} tests.')
    for role in ('calibration','diagnostic'):
        r=expected['roles'][role]
        print(f"{role.upper()} P95 DISAGREEMENT: {r['baseline']['p95_abs_disagreement_mps']:.3f} -> {r['candidate']['p95_abs_disagreement_mps']:.3f} m/s")
    print('OVERALL TARGET:', 'MET ON REUSED DEVELOPMENT DATA ONLY' if expected['candidate_provisional_target_met'] else 'NOT MET')
    print('WITNESS ENABLED: NO')

if __name__=='__main__':audit()
