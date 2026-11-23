"""Small stdlib helpers shared by experiment drivers."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def api(base, route, payload=None, timeout=1800):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(base.rstrip("/") + route, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        if not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            # SGLang's control endpoints return plain-text acknowledgements.
            # Generation/info endpoints must remain strict JSON evidence.
            if route in ("/flush_cache", "/start_profile", "/stop_profile"):
                return body.decode()
            raise


def gpu_sample():
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=5)
        if p.returncode:
            return {"at": now(), "error": p.stderr.strip() or p.stdout.strip()}
        return {"at": now(), "gpus": [dict(zip(
            ("index", "used_mib", "free_mib", "utilization_percent"),
            (int(x.strip()) for x in row.split(",")))) for row in p.stdout.splitlines()]}
    except (OSError, ValueError, subprocess.TimeoutExpired) as e:
        return {"at": now(), "error": str(e)}


def fields(value, key):
    """Find fields in nested/version-dependent server-info responses."""
    found = []
    if isinstance(value, dict):
        if key in value:
            found.append(value[key])
        for child in value.values():
            found.extend(fields(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(fields(child, key))
    return found
