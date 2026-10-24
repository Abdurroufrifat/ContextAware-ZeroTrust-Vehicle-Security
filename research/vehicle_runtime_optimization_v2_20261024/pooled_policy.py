"""Per-database durable connection reuse; every original transaction still commits."""
from runtime_common import *
from fast_policy import DurableWALPolicy
import os,sqlite3,threading
_registry={};_native=None;_auth=None
def key(path):return os.path.normcase(str(Path(path).absolute()))
class Loan:
 def __init__(self,owner):self.owner=owner
 def __getattr__(self,name):return getattr(self.owner._keeper,name)
 def close(self):
  p=self.owner
  if not p._borrowed:return
  if p._keeper.in_transaction:
   p._keeper.rollback();p._borrowed=False
   raise ValueError("Unfinished replay-state transaction")
  p._borrowed=False
def route(path):
 p=_registry.get(key(path))
 if p is None:return _native(path)
 p._own()
 safe=_auth.safe(path)
 if not safe.is_file():raise ValueError("Replay state missing; explicit provisioning required")
 st=safe.stat()
 if (st.st_dev,st.st_ino)!=p._file_identity:raise ValueError("Replay state file replaced")
 if p._borrowed:raise ValueError("Nested or concurrent replay-state connection")
 p._borrowed=True;return p._loan
class PooledPolicy(DurableWALPolicy):
 def __init__(self,*args,direct_forest=False,**kwargs):
  super().__init__(*args,direct_forest=direct_forest,**kwargs)
  global _native,_auth
  guard=self.source.inner.estimator.base.gateway.commands.report_guard
  module=sys.modules[type(guard).__module__]
  if _registry and module is not _auth:raise ValueError("Unexpected authentication module")
  self._borrowed=False;self._loan=Loan(self);st=self.database.stat();self._file_identity=(st.st_dev,st.st_ino);self._registered=False
  if not _registry:
   _auth=module;_native=module.connect;module.connect=route
  identity=key(self.database)
  if identity in _registry:raise ValueError("Replay database already owned")
  _registry[identity]=self;self._registered=True
 def close(self):
  if self._closed:return
  self._own()
  if self._borrowed:raise ValueError("Cannot close during replay-state transaction")
  if self._registered:
   del _registry[key(self.database)];self._registered=False
   if not _registry:
    if _auth.connect is not route:raise ValueError("Authentication connection dispatcher changed")
    _auth.connect=_native
  super().close()
class PooledFastPolicy(PooledPolicy):
 def __init__(self,*args,**kwargs):super().__init__(*args,direct_forest=True,**kwargs)
