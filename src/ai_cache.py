"""Content-addressed AI checkpoints, independent of transient image paths."""
import hashlib
import json
import os
from pathlib import Path
import pandas as pd


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_cache(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.pending")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def read_cache(path, columns):
    if not Path(path).exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path, low_memory=False, dtype={c: "string" for c in columns if c.endswith("_id") or c == "BUILDINGSTRUCTUREID"})


def vision_checkpoint_matches(cached, context_hash, current_image_hashes):
    if cached is None or str(cached.get('Context_SHA256','')) != context_hash:
        return False
    if cached.get('ai_decision') not in ['keep','merge','delete']:
        return False
    stored = {key:str(cached.get(key,'')) for key in ['plain_sha256','overlay_sha256']}
    if any(len(value) != 64 for value in stored.values()):
        return False
    if cached.get('Input_SHA256') != fingerprint({'context':context_hash,**stored}):
        return False
    return not current_image_hashes or all(stored[key] == value for key,value in current_image_hashes.items())
