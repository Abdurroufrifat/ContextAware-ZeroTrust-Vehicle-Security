"""Safely unpack the verified development archive; inspect selected numeric streams."""
import dataclasses, hashlib, json, tarfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "acquisition_manifest.json").read_text())
archive = ROOT / "raw_data" / manifest["recording"]
if hashlib.file_digest(archive.open("rb"), "sha256").hexdigest() != manifest["archive_sha256"]:
    raise RuntimeError("Archive changed since verification")
bag = ROOT / "raw_data" / "development_07_turning.bag"
if not bag.exists():
    print("Unpacking verified development bag", flush=True)
    with tarfile.open(archive, "r|xz") as stream:
        count = 0
        for member in stream:
            if not member.isfile() or not member.name.endswith(".bag"):
                continue
            if count or not (0 < member.size < 20_000_000_000):
                raise RuntimeError("Unexpected archive bag count/size")
            if any(x in member.name.lower() for x in ("urban02", "rav4", "final_evaluation")):
                raise RuntimeError("Reserved source excluded")
            with stream.extractfile(member) as inp, bag.with_suffix(".part").open("wb") as out:
                import shutil
                shutil.copyfileobj(inp, out, 4 * 1024 * 1024)
            if bag.with_suffix(".part").stat().st_size != member.size:
                raise RuntimeError("Truncated bag")
            count += 1
        if count != 1:
            raise RuntimeError("Expected exactly one development bag")
    bag.with_suffix(".part").rename(bag)
from rosbags.highlevel import AnyReader
def plain(value):
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if dataclasses.is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value) if not f.name.isupper()}
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
with AnyReader([bag]) as reader:
    topics = [dict(topic=c.topic, msgtype=c.msgtype, count=c.msgcount) for c in reader.connections]
    selected = [c for c in reader.connections if c.topic.lstrip("/") in {
        "actuator/vehicle/feedback", "localization/egomotion/odom",
        "sensor/ins/oxts_rt3000/gps/odom", "sensor/ins/oxts_rt3000/gps/fix",
        "sensor/ins/oxts_rt3000/imu/data", "tf_static"}]
    examples = {}
    for c, timestamp, raw in reader.messages(connections=selected):
        if c.topic not in examples:
            examples[c.topic] = dict(bag_time_ns=timestamp, message=plain(reader.deserialize(raw, c.msgtype)))
    result = dict(bag_bytes=bag.stat().st_size, topics=topics, selected_examples=examples)
    (ROOT / "bag_inventory.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(topics=[x for x in topics if any(c.topic==x["topic"] for c in selected)],
        examples=examples), indent=2), flush=True)
