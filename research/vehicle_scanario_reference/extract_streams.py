"""Extract selected ego numerical streams, preserving receipt and sensor timestamps."""
import dataclasses, hashlib, json
from collections import Counter
from pathlib import Path
from rosbags.highlevel import AnyReader
ROOT = Path(__file__).resolve().parent
BAG = ROOT / "raw_data" / "development_07_turning.bag"
TOPICS = {"actuator/vehicle/feedback", "localization/egomotion/odom",
    "sensor/ins/oxts_rt3000/gps/odom", "sensor/ins/oxts_rt3000/gps/fix",
    "sensor/ins/oxts_rt3000/imu/data", "tf_static"}
def plain(value):
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if dataclasses.is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value) if not f.name.isupper()}
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
def run():
    outdir = ROOT / "raw_data" / "extracted"
    outdir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    definitions = {}
    with AnyReader([BAG]) as reader, (outdir / "ego_streams.jsonl").open("w") as out:
        selected = [c for c in reader.connections if c.topic.lstrip("/") in TOPICS]
        for c in selected:
            definitions[c.topic] = dict(msgtype=c.msgtype, definition=str(c.msgdef))
        for connection, timestamp, raw in reader.messages(connections=selected):
            msg = plain(reader.deserialize(raw, connection.msgtype))
            topic = connection.topic.lstrip("/")
            out.write(json.dumps(dict(topic=topic, receipt_ns=timestamp, message=msg), allow_nan=False)+"\n")
            counts[topic] += 1
    (ROOT / "message_definitions.json").write_text(json.dumps(definitions, indent=2)+"\n")
    digest = hashlib.file_digest((outdir / "ego_streams.jsonl").open("rb"), "sha256").hexdigest()
    summary = dict(topic_counts=counts, extracted_sha256=digest, development_only=True,
        source_revision="c131d445f3ea51c9db6ed3e63c64eac762f01d6e")
    (ROOT / "extraction_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)
if __name__ == "__main__":
    run()
