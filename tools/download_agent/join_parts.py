"""Join parts made by the agent's split step, check SHA-256, un-gzip if needed.

    python join_parts.py <manifest .parts.json> <destination file>
"""
import hashlib
import json
import sys
import zlib
from pathlib import Path

man_path, dest = Path(sys.argv[1]), Path(sys.argv[2])
m = json.loads(man_path.read_text(encoding="utf-8"))
folder = man_path.parent
missing = [p["part"] for p in m["parts"] if not (folder / p["part"]).exists()]
if missing:
    sys.exit(f"missing parts: {missing}")
for p in m["parts"]:
    got = hashlib.sha256((folder / p["part"]).read_bytes()).hexdigest()
    if got != p["sha256"]:
        sys.exit(f"part {p['part']} damaged (sha mismatch)")
gz = m.get("compress") == "gzip"
d = zlib.decompressobj(31) if gz else None
h = hashlib.sha256()
n = 0
tmp = dest.with_name(dest.name + ".tmp")
dest.parent.mkdir(parents=True, exist_ok=True)
with open(tmp, "wb") as out:
    for p in m["parts"]:
        buf = (folder / p["part"]).read_bytes()
        if gz:
            buf = d.decompress(buf)
        out.write(buf)
        h.update(buf)
        n += len(buf)
    if gz:
        tail = d.flush()
        out.write(tail)
        h.update(tail)
        n += len(tail)
if n != m["bytes"] or h.hexdigest() != m["sha256"]:
    tmp.unlink()
    sys.exit(f"joined file does not match the manifest ({n} vs {m['bytes']} bytes)")
tmp.replace(dest)
print(f"ok {dest} {n} bytes from {len(m['parts'])} parts" + (" (gzip)" if gz else ""))
