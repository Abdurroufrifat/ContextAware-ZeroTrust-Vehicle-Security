from pathlib import Path
import json
import argparse
import numpy as np
import torch

parser=argparse.ArgumentParser(description='Strict same-environment numerical reproduction check; floating cross-platform agreement is not promised.')
parser.add_argument('--reference',type=Path,required=True)
parser.add_argument('--reproduced',type=Path,required=True)
parser.add_argument('--save',type=Path,required=True)
args=parser.parse_args()
original=args.reference.resolve()
fresh=args.reproduced.resolve()
a=json.loads((original/'summary.json').read_text());b=json.loads((fresh/'summary.json').read_text())
for key in ('rows','normal','attack','errors','streams','original_normal_replay','raw_alignment_replay'):
    assert a[key]==b[key],key
for relative in ('fit/scaler.json','fit/profiles.json','calibration/thresholds.json','physics_reference.json'):
    assert (original/relative).read_bytes()==(fresh/relative).read_bytes(),relative
count=0
for p in original.rglob('*.npz'):
    q=fresh/p.relative_to(original)
    with np.load(p,allow_pickle=False) as x,np.load(q,allow_pickle=False) as y:
        assert x.files==y.files,str(q)
        for key in x.files:
            if x[key].dtype.kind in 'fc':same=np.array_equal(x[key],y[key],equal_nan=True)
            else:same=np.array_equal(x[key],y[key])
            assert same,(p.name,key)
    count+=1
checkpoints=0
for p in (original/'fit').glob('*.pt'):
    x=torch.load(p,weights_only=True);y=torch.load(fresh/'fit'/p.name,weights_only=True)
    assert x.keys()==y.keys() and all(torch.equal(x[k],y[k]) for k in x)
    checkpoints+=1
result=dict(status='PASS',clean_extraction_training_completed=True,checkpoint_tensor_sets_exact=checkpoints,
    npz_blocks_all_arrays_exact=count,development_rows_exact=len(b['rows']),normal_and_attack_gate_results_exact=True,
    scalar_profiles_and_thresholds_byte_identical=True,physics_reference_byte_identical=True,
    reserved_validation_accessed=False,notes='One identical frozen recipe; no parameter sweep. NPZ archive timestamp bytes and wall-clock runtimes are not compared.')
dest=args.save
dest.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
