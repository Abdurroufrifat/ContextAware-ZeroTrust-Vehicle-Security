"""Opt-in sequential runtime adapter. Frozen policies and models are unchanged."""
from runtime_common import Original
from pathlib import Path
import sqlite3,os,threading
import numpy as np
from sklearn.ensemble import RandomForestRegressor
class DirectForest:
    """Same float32 splits, tree order and float64 sum as single-output sklearn RF."""
    def __init__(self,model):
        if not isinstance(model,RandomForestRegressor) or model.n_outputs_!=1:raise ValueError("Expected frozen single-output forest")
        self.original=model;self.n_features_in_=model.n_features_in_;self.trees=tuple(e.tree_ for e in model.estimators_)
    def predict(self,X):
        try:x=np.asarray(X,dtype=np.float32,order="C")
        except (ValueError,TypeError):return self.original.predict(X)
        if x.ndim!=2 or x.shape[1]!=self.n_features_in_ or x.shape[0]==0 or not np.isfinite(x).all():
            return self.original.predict(X)
        out=np.zeros(x.shape[0],dtype=np.float64)
        for i in range(len(x)):
            for tree in self.trees:
                node=0
                while tree.children_left[node]!=-1:
                    node=tree.children_left[node] if float(x[i,tree.feature[node]])<=float(tree.threshold[node]) else tree.children_right[node]
                out[i]+=tree.value[node,0,0]
        out/=len(self.trees)
        return out
class DurableWALPolicy(Original):
    """One owner thread, one writer sequence; every original FULL commit is retained."""
    def __init__(self,state_directory,context,keys,method="joint",*,direct_forest=False):
        if method!="joint":raise ValueError("Runtime adapter is qualified only for joint policy")
        super().__init__(state_directory,context,keys,method)
        self._owner=(os.getpid(),threading.get_ident());self._closed=False
        self.database=Path(state_directory)/"replay.sqlite"
        self._keeper=sqlite3.connect(self.database.as_uri()+"?mode=rw",uri=True,timeout=10,isolation_level=None)
        try:
            mode=self._keeper.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if mode.lower()!="wal":raise ValueError("WAL unavailable")
            self._keeper.execute("PRAGMA synchronous=FULL")
            self._keeper.execute("PRAGMA wal_autocheckpoint=1000")
            if self._keeper.execute("PRAGMA synchronous").fetchone()[0]!=2:raise ValueError("FULL sync unavailable")
            self._keeper.execute("SELECT name,value FROM metadata").fetchall()
            self.direct_forest=direct_forest
            if direct_forest:
                guard=self.source.inner.estimator.refined
                guard.model=DirectForest(guard.model)
        except BaseException:
            self._keeper.close();self._closed=True;raise
    def _own(self):
        if self._closed:raise ValueError("Runtime adapter is closed")
        if self._owner!=(os.getpid(),threading.get_ident()):raise ValueError("Single owner thread/process required")
    def process(self,record):
        self._own();return super().process(record)
    def command_packet(self,*args,**kwargs):
        self._own();return super().command_packet(*args,**kwargs)
    def actuate(self,*args,**kwargs):
        self._own();return super().actuate(*args,**kwargs)
    def watchdog(self,now):
        self._own();return super().watchdog(now)
    def close(self):
        if self._closed:return
        self._own()
        try:
            checkpoint=self._keeper.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint[0]!=0:raise ValueError("Checkpoint busy")
        finally:self._keeper.close();self._closed=True
class FastLifecyclePolicy(DurableWALPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs,direct_forest=True)
