"""Opt-in local durable replay state for the fixed VehiclePolicyEngine interface.

All instances must share one trusted, explicitly provisioned local database.
Deletion, cloning, valid older snapshots, compromised keys, and external service
actions are outside this adapter's guarantees. Storage errors return no verdict.
"""
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
import hashlib
import hmac
import json
import os
import sqlite3
import sys
from .policy import (SOURCES, PolicyDecision, VehiclePolicyEngine, VehicleRequest,
                     canonical_json, digest, _integer, _text, _unique_pairs, _nonfinite)

class DurableStateError(RuntimeError):
    """No authorization result is available; the caller must take no action."""

class DurableVehiclePolicy:
    """Serialize load -> base policy operation -> save -> COMMIT before return.

    Coupling to base-private state is intentional and bound to its source digest.
    This is a research adapter, not automatic replacement of existing policies.
    """
    def __init__(self,path,source_keys,capability_key,permissions,config=None):
        try: self.path=Path(path).resolve()
        except (OSError,UnicodeError) as exc:
            raise DurableStateError('database_path_unavailable') from exc
        self._keys=dict(source_keys)
        self._capability_key=capability_key
        self._permissions=frozenset(permissions)
        probe=VehiclePolicyEngine(self._keys,capability_key,self._permissions,config)
        self._config=probe.config
        self._binding=digest(dict(version='coact-durable-state-v1',base_policy=probe.policy_digest,
                                  adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                  collector_keys={s:hashlib.sha256(k).hexdigest() for s,k in self._keys.items()}))
        self._state_key=hmac.new(capability_key,b'COACT-DURABLE-STATE-KEY-V1\0',hashlib.sha256).digest()

    @property
    def policy_digest(self): return self._binding

    @property
    def config(self): return self._config

    def _engine(self):
        e=VehiclePolicyEngine(self._keys,self._capability_key,self._permissions,self._config)
        e._policy_digest=self._binding
        return e

    @staticmethod
    def _cleanup(c):
        # Preserve any primary failure; otherwise a failed close cancels a
        # pending successful return so no grant escapes the storage contract.
        primary_in_flight=sys.exc_info()[0] is not None
        failure=None
        try:
            if c.in_transaction: c.rollback()
        except sqlite3.Error as exc: failure=exc
        try: c.close()
        except sqlite3.Error as exc:
            if failure is None: failure=exc
        if failure is not None and not primary_in_flight:
            raise DurableStateError('durable_state_cleanup_failed_no_authorization') from failure

    def _connect(self):
        # mode=rw prevents a missing file from silently becoming empty state.
        c=sqlite3.connect(self.path.as_uri()+'?mode=rw',uri=True,timeout=1.0,isolation_level=None)
        try:
            if c.execute('PRAGMA journal_mode').fetchone()[0]!='delete':
                raise DurableStateError('unsupported_journal_mode')
            c.execute('PRAGMA synchronous=EXTRA')
            if c.execute('PRAGMA synchronous').fetchone()[0]!=3:
                raise DurableStateError('required_sync_mode_unavailable')
            return c
        except Exception:
            self._cleanup(c)
            raise

    def _mac(self,payload):
        return hmac.new(self._state_key,b'COACT-DURABLE-STATE-V1\0'+self._binding.encode('ascii')+b'\0'+payload.encode('ascii'),hashlib.sha256).hexdigest()

    def _dump(self,e):
        return canonical_json(dict(version=1,clock=e._clock_watermark,
                                   watermarks=[[v,s,n] for (v,s),n in sorted(e._watermarks.items())],
                                   nonces=[[v,n,t] for (v,n),t in sorted(e._used_nonces.items())],
                                   decisions=[[key,asdict(d)] for key,d in e._decisions.items()]))

    def _load(self,c):
        rows=c.execute('SELECT id,binding,payload,mac FROM policy_state').fetchall()
        if len(rows)!=1 or rows[0][0]!=1: raise DurableStateError('missing_or_ambiguous_state')
        _,binding,payload,mac=rows[0]
        if binding!=self._binding or type(payload) is not str or type(mac) is not str:
            raise DurableStateError('state_policy_or_key_binding_mismatch')
        try:
            if not hmac.compare_digest(mac,self._mac(payload)): raise DurableStateError('state_authentication_failed')
            state=json.loads(payload,object_pairs_hook=_unique_pairs,parse_constant=_nonfinite)
            if set(state)!={'version','clock','watermarks','nonces','decisions'} or type(state['version']) is not int or state['version']!=1:
                raise ValueError('invalid_state_schema')
            if type(state['clock']) is not int or not -1<=state['clock']<=2**53-1:
                raise ValueError('invalid_clock_watermark')
            e=self._engine();e._clock_watermark=state['clock']
            for field,destination in [('watermarks',e._watermarks),('nonces',e._used_nonces)]:
                if type(state[field]) is not list: raise ValueError('invalid_state_list')
                for row in state[field]:
                    if type(row) is not list or len(row)!=3: raise ValueError('invalid_state_row')
                    vehicle,name,value=row
                    if not _text(vehicle) or not _text(name) or not _integer(value) or (vehicle,name) in destination:
                        raise ValueError('invalid_state_key_or_value')
                    if field=='watermarks' and name not in SOURCES: raise ValueError('unknown_watermark_source')
                    destination[vehicle,name]=value
            if type(state['decisions']) is not list or len(state['decisions'])>4096: raise ValueError('invalid_decision_registry')
            e._decisions=OrderedDict()
            for key,fields in state['decisions']:
                values=dict(fields);values['request']=VehicleRequest(**values['request'])
                values['reasons']=tuple(values['reasons']);values['components']=tuple(tuple(v) for v in values['components'])
                d=PolicyDecision(**values)
                if key in e._decisions or key!=digest(asdict(d)) or d.policy_digest!=self._binding:
                    raise ValueError('invalid_registered_decision')
                e._decisions[key]=d
            return e
        except (ValueError,TypeError,KeyError,UnicodeError,OverflowError,RecursionError) as exc:
            raise DurableStateError('invalid_authenticated_state') from exc

    @classmethod
    def provision(cls,path,source_keys,capability_key,permissions,config=None):
        obj=cls(path,source_keys,capability_key,permissions,config)
        c=None
        try:
            # Trusted one-time enrollment only; never truncate existing history.
            fd=os.open(obj.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
            c=obj._connect();c.execute('BEGIN IMMEDIATE')
            c.execute('CREATE TABLE policy_state(id INTEGER PRIMARY KEY CHECK(id=1),binding TEXT NOT NULL,payload TEXT NOT NULL,mac TEXT NOT NULL)')
            payload=obj._dump(obj._engine())
            c.execute('INSERT INTO policy_state VALUES(1,?,?,?)',(obj._binding,payload,obj._mac(payload)))
            c.execute('COMMIT')
            return obj
        except (sqlite3.Error,OSError,UnicodeError) as exc:
            raise DurableStateError('explicit_state_provisioning_failed') from exc
        finally:
            if c is not None: obj._cleanup(c)

    def _run(self,method,*args):
        c=None
        try:
            c=self._connect();c.execute('BEGIN IMMEDIATE')
            e=self._load(c)
            result=getattr(e,method)(*args)
            payload=self._dump(e)
            changed=c.execute('UPDATE policy_state SET payload=?,mac=? WHERE id=1 AND binding=?',(payload,self._mac(payload),self._binding))
            if changed.rowcount!=1: raise DurableStateError('state_update_failed')
            c.execute('COMMIT')
            return result
        except (sqlite3.Error,OSError,UnicodeError) as exc:
            raise DurableStateError('durable_state_unavailable_no_authorization') from exc
        finally:
            if c is not None: self._cleanup(c)

    def evaluate(self,request,frames,now_ms):
        return self._run('evaluate',request,frames,now_ms)

    def issue_capability(self,decision):
        return self._run('issue_capability',decision)

    def consume_capability(self,capability,request,frames,now_ms):
        return self._run('consume_capability',capability,request,frames,now_ms)
