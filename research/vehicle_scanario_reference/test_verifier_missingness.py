import contextlib,csv,hashlib,io,json,math,tempfile,unittest
from pathlib import Path
import verify_pilot
class VerifierMissingnessTests(unittest.TestCase):
    def run_fixture(self,reported_wheel,rear_x=-1.4275,yaw_rate=0.):
        parent=Path(__file__).resolve().parent/"raw_data"
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            root=Path(tmp)
            folder=root/"raw_data"/"extracted"
            folder.mkdir(parents=True)
            header=lambda t:dict(stamp=dict(sec=0,nanosec=t))
            refs=dict(header=header(100),twist=dict(twist=dict(linear=dict(x=3.,y=0.,z=0.),angular=dict(x=0.,y=0.,z=yaw_rate))))
            feedback=dict(header=header(90),wheel_speed=[dict(id=i,speed=dict(mean=4.,availability=dict(data=1))) for i in (1,2,3,4)])
            rows=[dict(topic="sensor/ins/oxts_rt3000/gps/odom",receipt_ns=100,message=refs),
                  dict(topic="actuator/vehicle/feedback",receipt_ns=101,message=feedback)]
            transforms=[dict(header=dict(frame_id="base_link"),child_frame_id=name,
                transform=dict(translation=dict(x=x,y=0.,z=z),rotation=dict(x=0.,y=0.,z=0.,w=1.)))
                for name,x,z in [("sensor/ins/oxts",0.,0.),("vehicle/rear_axis",rear_x,-.2715)]]
            rows.insert(0,dict(topic="tf_static",receipt_ns=0,message=dict(transforms=transforms)))
            source=folder/"ego_streams.jsonl"
            source.write_text("".join(json.dumps(r)+"\n" for r in rows))
            (root/"extraction_summary.json").write_text(json.dumps(dict(extracted_sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
            summary=dict(queries=1,comparisons=dict(wheel_median_mps=dict(absolute_discrepancy_mps=dict(p95=None))),
                         physical_gate_pass=None,policy_promotion_allowed=False)
            (root/"pilot_summary.json").write_text(json.dumps(summary))
            with (folder/"development_pairs.csv").open("w",newline="") as out:
                writer=csv.DictWriter(out,fieldnames=["receipt_ns","sensor_ns","wheel_median_mps","wheel_age_ns","ins_rear_axis_speed_mps"])
                writer.writeheader()
                writer.writerow(dict(receipt_ns=100,sensor_ns=100,wheel_median_mps=reported_wheel,
                                     wheel_age_ns="" if reported_wheel=="" else 10,ins_rear_axis_speed_mps=math.hypot(3.,rear_x*yaw_rate)))
            previous=verify_pilot.ROOT
            try:
                verify_pilot.ROOT=root
                with contextlib.redirect_stdout(io.StringIO()): verify_pilot.run()
            finally: verify_pilot.ROOT=previous
    def test_causal_abstention_is_verifiable(self):
        self.run_fixture("")
    def test_published_offset_is_read_instead_of_hardcoded(self):
        self.run_fixture("",rear_x=-2.,yaw_rate=1.)
    def test_future_wheel_cannot_be_reported_as_available(self):
        with self.assertRaises(AssertionError): self.run_fixture(4.)
if __name__=="__main__": unittest.main()
