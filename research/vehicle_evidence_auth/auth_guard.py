"""Research-only authenticated-report guard. Authentication never grants speed trust.
No key provisioning, sensor attestation or protected storage is supplied here.
"""
from dataclasses import dataclass,field
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sqlite3

FIELDS=('schema','source_id','key_id','vehicle_id','session_id','capture_id','channel','seq','measured_us','speed_mmps')
IDS=('source_id','key_id','vehicle_id','session_id','capture_id','channel')
IDENTIFIER=re.compile(r'[A-Za-z0-9_.:-]{1,64}\Z')
TAG=re.compile(r'[0-9a-f]{64}\Z')
DOMAIN=b'COACTCPS-EVIDENCE-HMAC-SHA256-v1\x00'
MAX_AGE_US=100000


@dataclass(frozen=True)
class Context:
    vehicle_id:str
    session_id:str
    capture_id:str


@dataclass(frozen=True)
class KeyRule:
    source_id:str
    key_id:str
    channel:str
    key:bytes=field(repr=False)
    enabled:bool=True


def identifier(value):
    return type(value) is str and IDENTIFIER.fullmatch(value) is not None


def unsigned(value):
    return type(value) is int and 0<=value<2**63


def context_text(context):
    if not isinstance(context,Context) or not all(identifier(v) for v in (context.vehicle_id,context.session_id,context.capture_id)):
        raise ValueError('Invalid provisioned context')
    return json.dumps([context.vehicle_id,context.session_id,context.capture_id],separators=(',',':'))


def safe(path):
    p=Path(path)
    for text in (str(p.absolute()).lower(),str(p.resolve()).lower()):
        if any(s in text for s in ('urban02','rav4','vehicle_final_evaluation')):
            raise ValueError('Protected location refused')
    if p.is_symlink():raise ValueError('Replay state symlink refused')
    return p.resolve()


def key_check(key):
    if type(key) is not bytes or len(key)<32:
        raise ValueError('A key must be at least32bytes; actual provisioning is external')


def canonical(body):
    if type(body) is not dict or set(body)!=set(FIELDS):raise ValueError('Unexpected body schema')
    if type(body['schema']) is not int or body['schema']!=1:raise ValueError('Invalid schema version')
    if not all(identifier(body[k]) for k in IDS):raise ValueError('Invalid identifier')
    if not all(unsigned(body[k]) for k in ('seq','measured_us','speed_mmps')):raise ValueError('Invalid integer field')
    return DOMAIN+json.dumps([body[k] for k in FIELDS],ensure_ascii=True,separators=(',',':')).encode('ascii')


def sign(body,key):
    key_check(key)
    return dict(body=dict(body),tag=hmac.new(key,canonical(body),hashlib.sha256).hexdigest())


def parse_wire(raw):
    if type(raw) not in (str,bytes) or len(raw.encode('utf-8') if isinstance(raw,str) else raw)>4096:
        raise ValueError('Invalid wire size or type')
    def unique(pairs):
        obj={}
        for k,v in pairs:
            if k in obj:raise ValueError('Duplicate JSON key')
            obj[k]=v
        return obj
    try:
        result=json.loads(raw,object_pairs_hook=unique)
        validate_envelope(result)
        return result
    except (ValueError,UnicodeError,TypeError,RecursionError) as e:
        raise ValueError('Invalid envelope encoding') from e


def validate_envelope(message):
    if type(message) is not dict or set(message)!= {'body','tag'}:
        raise ValueError('Invalid envelope shape')
    canonical(message['body'])
    if type(message['tag']) is not str or TAG.fullmatch(message['tag']) is None:
        raise ValueError('Invalid tag encoding')


def connect(path):
    p=safe(path)
    if not p.is_file():raise ValueError('Replay state missing; explicit provisioning required')
    try:
        return sqlite3.connect(p.as_uri()+'?mode=rw',uri=True,timeout=10,isolation_level=None)
    except sqlite3.Error as e:
        raise ValueError('Replay state unavailable') from e


def read_context(connection,context):
    try:
        data=dict(connection.execute('SELECT name,value FROM metadata'))
        if data.get('schema')!='1' or data.get('context')!=context_text(context):
            raise ValueError('Replay state schema/context mismatch')
        last=int(data['last_now_us'])
        if last< -1 or last>=2**63:raise ValueError('Invalid stored clock')
        return last
    except (sqlite3.Error,KeyError,TypeError) as e:
        raise ValueError('Invalid replay state') from e


def create_state(path,context):
    p=safe(path);text=context_text(context)
    p.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    connection=None
    try:
        connection=connect(p)
        connection.execute('PRAGMA synchronous=FULL')
        connection.execute('BEGIN IMMEDIATE')
        connection.execute('CREATE TABLE metadata(name TEXT PRIMARY KEY,value TEXT NOT NULL)')
        connection.execute('CREATE TABLE counters(source_id TEXT NOT NULL,channel TEXT NOT NULL,seq INTEGER NOT NULL,PRIMARY KEY(source_id,channel))')
        connection.executemany('INSERT INTO metadata VALUES (?,?)',[('schema','1'),('context',text),('last_now_us','-1')])
        connection.commit()
    except Exception:
        if connection is not None:connection.rollback();connection.close();connection=None
        p.unlink(missing_ok=True)
        raise
    finally:
        if connection is not None:connection.close()


def result(status,mac_valid=False):
    return dict(status=status,mac_valid=mac_valid,authenticated_report=status=='authenticated',
                speed_authorized=False,measurement_truth_established=False,
                physical_witness_enabled=False,promotion_allowed=False)


class Guard:
    def __init__(self,path,context,rules):
        self.path=safe(path);self.context=context;self.rules={};seen_keys={}
        context_text(context)
        for rule in rules:
            if not isinstance(rule,KeyRule) or not all(identifier(v) for v in (rule.source_id,rule.key_id,rule.channel)) or type(rule.enabled) is not bool:
                raise ValueError('Invalid allowlist rule')
            key_check(rule.key)
            identity=(rule.source_id,rule.key_id)
            if identity in self.rules:raise ValueError('Duplicate source/key rule')
            if rule.key in seen_keys and seen_keys[rule.key]!=rule.source_id:
                raise ValueError('Different sources must not share a key')
            seen_keys[rule.key]=rule.source_id;self.rules[identity]=rule
        connection=connect(self.path)
        try:read_context(connection,self.context)
        finally:connection.close()

    def verify(self,message,now_us,received_us):
        try:validate_envelope(message)
        except (ValueError,TypeError):return result('malformed')
        body=message['body'];rule=self.rules.get((body['source_id'],body['key_id']))
        if rule is None:return result('unknown_key')
        if not rule.enabled:return result('revoked_key')
        expected=hmac.new(rule.key,canonical(body),hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected,message['tag']):return result('bad_tag')
        if (body['vehicle_id'],body['session_id'],body['capture_id'])!=(self.context.vehicle_id,self.context.session_id,self.context.capture_id):
            return result('context_mismatch',True)
        if body['channel']!=rule.channel:return result('channel_denied',True)
        measured=body['measured_us']
        if not unsigned(now_us) or not unsigned(received_us) or not measured<=received_us<=now_us:
            return result('invalid_clock_order',True)
        if now_us-received_us>MAX_AGE_US:return result('receipt_stale',True)
        if now_us-measured>MAX_AGE_US:return result('measurement_stale',True)
        connection=connect(self.path)
        try:
            connection.execute('PRAGMA synchronous=FULL')
            connection.execute('BEGIN IMMEDIATE')
            last=read_context(connection,self.context)
            if now_us<last:
                connection.rollback();return result('clock_rollback',True)
            saved=connection.execute('SELECT seq FROM counters WHERE source_id=? AND channel=?',(body['source_id'],body['channel'])).fetchone()
            if saved is not None and body['seq']<=saved[0]:
                connection.rollback();return result('replay_or_reorder',True)
            connection.execute('INSERT INTO counters VALUES (?,?,?) ON CONFLICT(source_id,channel) DO UPDATE SET seq=excluded.seq',(body['source_id'],body['channel'],body['seq']))
            connection.execute("UPDATE metadata SET value=? WHERE name='last_now_us'",(str(now_us),))
            connection.commit()
            return result('authenticated',True)
        except sqlite3.Error as e:
            connection.rollback();raise ValueError('Replay-state transaction failed') from e
        finally:connection.close()
