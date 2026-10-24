from paired_core import *
def main():
 freeze();target=ROOT/'results.json'
 if target.exists():raise ValueError('paired outcomes already generated; do not overwrite')
 members,y,pred,support=load_inputs();r=build(members,y,pred,support);write(target,r)
 for n in COMPARATORS:
  print(n,'window P95 wins/ties/losses for original:',r['unit_metric_wins']['window'][n]['p95_mps'],flush=True)
 print(r['decision'],flush=True)
if __name__=='__main__':main()
