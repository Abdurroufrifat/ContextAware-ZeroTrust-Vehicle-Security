"""Controlled exit of this test process only, retaining SQLite recovery artifacts."""
from runtime_common import *
from pooled_policy import PooledFastPolicy as FastLifecyclePolicy
import sqlite3,os
state=Path(sys.argv[1]);mode=sys.argv[2]
t=next(t for t in load(B/"results.json")["trials"] if t["method"]=="joint" and t["case"]=="nominal")
x=trace(t)[0];p=FastLifecyclePolicy(state,x["context"],KEYS,"joint")
out=p.process(x["input"])
if mode=="uncommitted":
 c=sqlite3.connect(p.database.as_uri()+"?mode=rw",uri=True,isolation_level=None)
 c.execute("PRAGMA synchronous=FULL");c.execute("BEGIN IMMEDIATE");c.execute("UPDATE counters SET seq=999")
print("COMMITTED_FIRST_FRAME",flush=True)
os._exit(0)
