"""Versioned lifecycle gate: preserve V1 development evidence; tighten actor/time contract."""
from packet_policy import LifecyclePolicy as V1
from certificate_v2 import verify_certificate
import json
class LifecyclePolicy(V1):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._active_until=None;self._active_issued=None;self._active_value=-4000;self._cadence_failed=False;self._cycle_complete=True
    def process(self,record):
        if type(record) is not dict or type(record.get("now_us")) is not int:raise ValueError("packet time")
        if not self._cycle_complete:
            self._cadence_failed=True
            raise ValueError("previous actuation cycle missing; watchdog backup and fresh bootstrap required")
        if self._cadence_failed or (self.last is not None and record["now_us"]-self.last!=100000):
            self._cadence_failed=True
            raise ValueError("control cadence broken; watchdog backup and fresh bootstrap session required")
        out=super().process(record)
        self._cycle_complete=False
        out["certificate"]["watchdog_period_us"]=100000
        out["certificate"]["delivery_contract"]="exact issued-time delivery; no latency tolerance"
        return out
    def actuate(self,packet,now):
        if type(now) is not int or now<0:return -4000,"invalid_clock"
        if self._cadence_failed or self.pending is None:return -4000,"cadence_or_evidence_missing"
        self._cycle_complete=True
        if type(packet) is not dict:
            self.accepted=-4000;self._active_value=-4000;self._active_issued=now;self._active_until=now
            return -4000,"missing_or_malformed"
        bare={k:packet[k] for k in ("wire","received_us") if k in packet}
        accepted,status=self.source.actuate(bare,now)
        if status=="authenticated":
            good,reason=verify_certificate(packet.get("certificate"),packet["wire"],self.keys["actuator"],self.context,now,self.pending["bounds"],self.error,self.pending["digest"],self.seen)
            if good:self.seen=json.loads(packet["wire"])["body"]["seq"]
            else:accepted=-4000;status="certificate_"+reason
        self.accepted=accepted
        self._active_value=accepted;self._active_issued=now;self._active_until=now+100000
        return accepted,status
    def watchdog(self,now):
        if type(now) is not int or now<0:raise ValueError("watchdog clock")
        if self._cadence_failed or self._active_until is None:return -4000,"watchdog_no_authority"
        if now<self._active_issued:raise ValueError("watchdog clock rewind")
        if now>=self._active_until:return -4000,"watchdog_expired"
        return self._active_value,"watchdog_active"
