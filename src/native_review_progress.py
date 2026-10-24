"""Dataset-bound local batch tracking for human/native-agent row review.

This module never calls a model or marks queued rows reviewed. A row counts only
after an exact, validated results file is ingested.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

GROUPS = {"identity_geometry", "dimensions", "areas_proportions", "function", "dates_provenance_model"}
OUTCOMES = {"contradiction_found", "partially_supported", "insufficient_evidence", "identity_ambiguous"}
MODEL = "gpt-6-luna"
KEY = "Building_ID_建筑ID"
WEB_MODE = "source_research"
NO_WEB_MODE = "attribute_consistency_only_no_web"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def binding(dataset_sha: str, bid: str) -> str:
    return hashlib.sha256((dataset_sha + ":" + bid).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_manifest(state: Path) -> dict:
    p = state / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {
        "format": 1, "batches": {}, "completed": {}, "legacy_imported": False}


def _load_dataset(path: Path):
    """Read only the requested records after the cheap ID-only scan."""
    import pandas as pd
    import pyarrow.parquet as pq
    import shapely
    from pyproj import CRS, Transformer

    dataset_sha = sha256(path)
    ids = _dataset_ids(path)
    metadata = pq.read_metadata(path).metadata or {}
    geo = json.loads(metadata[b"geo"]) if b"geo" in metadata else {}
    geom_col = "geometry" if "geometry" in geo.get("columns", {}) else None
    source_crs = geo.get("columns", {}).get(geom_col, {}).get("crs") if geom_col else None
    if geom_col and source_crs is None:
        raise ValueError("geometry has no declared CRS")
    crs = CRS.from_json_dict(source_crs) if isinstance(source_crs, dict) else CRS.from_user_input(source_crs) if source_crs else None
    transformer = Transformer.from_crs(crs, 4326, always_xy=True) if crs else None
    return dataset_sha, ids, (pq, pd, shapely, transformer, geom_col, crs.to_string() if crs else None)


def _dataset_ids(path: Path):
    import pandas as pd
    ids = pd.read_parquet(path, columns=[KEY])[KEY]
    if not ids.is_unique:
        raise ValueError(f"dataset must have unique {KEY}")
    return ids.astype(str).tolist()


def _load_records(path: Path, selected: list[str], digest: str, helpers):
    pq, pd, shapely, transformer, geom_col, crs_text = helpers
    table = pq.read_table(path, filters=[(KEY, "in", selected)])
    df = table.to_pandas()
    df[KEY] = df[KEY].astype(str)
    rows = []
    for _, s in df.iterrows():
        row = {k: (None if pd.isna(v) else v.item() if hasattr(v, "item") else v)
               for k, v in s.items() if k != geom_col}
        bid = row[KEY]
        meta = {"record_binding_sha256": binding(digest, bid), "dataset_sha256": digest}
        geom_hash = None
        if geom_col and s[geom_col] is not None and not pd.isna(s[geom_col]):
            geom = s[geom_col] if hasattr(s[geom_col], "wkb") else shapely.from_wkb(s[geom_col])
            pt = geom.representative_point()
            lon, lat = transformer.transform(pt.x, pt.y)
            geom_hash = hashlib.sha256(geom.wkb).hexdigest()
            meta.update({"geometry_crs": crs_text, "geometry_sha256": geom_hash,
                         "representative_longitude": lon, "representative_latitude": lat,
                         "geometry_area_sqm": geom.area, "geometry_valid": bool(geom.is_valid),
                         "geometry_bounds_m": list(geom.bounds)})
        fingerprint_value = {"public_attributes": row, "geometry_sha256": geom_hash, "geometry_crs": crs_text}
        meta["content_fingerprint_sha256"] = hashlib.sha256(json.dumps(
            fingerprint_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        row["_review"] = meta
        rows.append(row)
    indexed = {r[KEY]: r for r in rows}
    if set(indexed) != set(selected):
        raise ValueError("selected dataset IDs could not be loaded exactly")
    return [indexed[bid] for bid in selected]


def _snapshot_review_ids(state: Path, digest: str, ids: list[str], helpers, dataset: Path) -> tuple[set[str], set[str]]:
    manifest = read_manifest(state)
    exact = {bid for bid in ids if manifest["completed"].get(bid, {}).get("binding") == binding(digest, bid)}
    comparable = [bid for bid in ids if bid not in exact and manifest["completed"].get(bid, {}).get("content_fingerprint_sha256")]
    carried = set()
    if comparable:
        rows = _load_records(dataset, comparable, digest, helpers)
        carried = {row[KEY] for row in rows
                   if row["_review"]["content_fingerprint_sha256"] == manifest["completed"][row[KEY]]["content_fingerprint_sha256"]}
    return exact, carried


def current_reviewed_ids(dataset: Path, state: Path) -> tuple[set[str], set[str]]:
    """Return exact-snapshot and unchanged-carried reviewed IDs without hashing untouched rows."""
    digest, ids, helpers = _load_dataset(dataset)
    return _snapshot_review_ids(state, digest, ids, helpers, dataset)


def _inspection_packet(rows: list[dict], input_sha: str, digest: str) -> dict:
    """Lossless column factoring for every public value; compact geometry for inspection."""
    public_names = [name for name in rows[0] if name != "_review"]
    constants, columns, values = {}, [], []
    for name in public_names:
        first = rows[0][name]
        signature = json.dumps(first, ensure_ascii=False, sort_keys=True, default=str)
        if all(json.dumps(row[name], ensure_ascii=False, sort_keys=True, default=str) == signature for row in rows[1:]):
            constants[name] = first
        else:
            columns.append(name)
    for row in rows:
        values.append([row[name] for name in columns])
    geom_names = [name for name in rows[0]["_review"]
                  if name not in {"record_binding_sha256", "content_fingerprint_sha256", "dataset_sha256",
                                  "batch_id", "geometry_sha256"}]
    geom_constants, geom_columns, geom_values = {}, [], []
    for name in geom_names:
        first = rows[0]["_review"][name]
        signature = json.dumps(first, ensure_ascii=False, sort_keys=True, default=str)
        if all(json.dumps(row["_review"][name], ensure_ascii=False, sort_keys=True, default=str) == signature for row in rows[1:]):
            geom_constants[name] = first
        else:
            geom_columns.append(name)
    for row in rows:
        geom_values.append([row["_review"][name] for name in geom_columns])
    return {"format": 1, "dataset_sha256": digest, "full_input_sha256": input_sha,
            "ids": [str(row[KEY]) for row in rows], "public_constants": constants,
            "public_columns": columns, "public_rows": values,
            "geometry_constants": geom_constants, "geometry_columns": geom_columns,
            "geometry_rows": geom_values}


def _result_ids(results):
    ids = [str(r.get("building_id", "")) for r in results]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("results must have unique nonempty building_id values")
    return ids


def validate_results(results, expected: dict[str, str], require_queries: bool = True) -> list[dict]:
    if not isinstance(results, list):
        raise ValueError("results file must be a JSON array")
    ids = _result_ids(results)
    if set(ids) != set(expected) or len(ids) != len(expected):
        raise ValueError("results IDs must match the batch exactly")
    for r in results:
        bid = str(r["building_id"])
        if r.get("record_binding_sha256") != expected[bid]:
            raise ValueError(f"record binding mismatch for {bid}")
        if r.get("model") != MODEL:
            raise ValueError(f"unexpected model for {bid}")
        if r.get("outcome") not in OUTCOMES:
            raise ValueError(f"invalid outcome for {bid}")
        if not isinstance(r.get("findings"), list) or not r["findings"]:
            raise ValueError(f"missing findings for {bid}")
        if not isinstance(r.get("unverified_groups"), list) or not r["unverified_groups"]:
            raise ValueError(f"missing unverified_groups for {bid}")
        if not GROUPS.issubset(set(r.get("reviewed_field_groups", []))):
            raise ValueError(f"five field groups not reviewed for {bid}")
        if r.get("fact_verified") is True or r.get("fully_fact_verified") is True:
            raise ValueError(f"row review cannot claim full fact verification for {bid}")
        mode = r.get("review_mode")
        if require_queries and mode not in {WEB_MODE, NO_WEB_MODE}:
            raise ValueError(f"review_mode must be declared for {bid}")
        sources = r.get("sources")
        if not isinstance(sources, list):
            raise ValueError(f"sources must be a list for {bid}")
        for source in sources:
            if source.get("retrieval_status") not in {"opened_this_pass", "historical_reference", "historical_link_not_opened", "not_retrieved", "retrieval_failed"}:
                raise ValueError(f"unrecognized source retrieval status for {bid}")
            if source.get("retrieval_status") == "opened_this_pass" and not source.get("url"):
                raise ValueError(f"opened source missing URL for {bid}")
            if require_queries and mode == NO_WEB_MODE and source.get("retrieval_status") == "opened_this_pass":
                raise ValueError(f"no-web review cannot claim an opened source for {bid}")
        queries = r.get("queries_used")
        if require_queries and mode == NO_WEB_MODE:
            reason = r.get("query_not_run_reason")
            if queries != [] or not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"no-web review requires empty queries_used and query_not_run_reason for {bid}")
        if require_queries and mode == WEB_MODE and (not isinstance(queries, list) or not queries):
            raise ValueError(f"source research requires actual queries for {bid}")
        if (not isinstance(queries, list) or (require_queries and mode == WEB_MODE and not queries)
                or not all(isinstance(q, str) and q.strip() for q in queries)):
            raise ValueError(f"queries_used required for {bid}")
    return results


def export_batch(dataset: Path, state: Path, batch_id: str, limit: int, ids_file: Path | None = None) -> Path:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not batch_id or Path(batch_id).name != batch_id or batch_id in {".", ".."}:
        raise ValueError("batch-id must be a simple filename stem")
    manifest = read_manifest(state)
    if batch_id in manifest["batches"]:
        raise ValueError("batch ID already exists; batches are immutable")
    digest, dataset_ids, helpers = _load_dataset(dataset)
    id_set = set(dataset_ids)
    requested = None
    if ids_file:
        with ids_file.open(encoding="utf-8-sig", newline="") as f:
            requested = {r[0].strip() for r in csv.reader(f) if r and r[0].strip()}
    if requested is not None:
        missing = requested - id_set
        if missing:
            raise ValueError(f"requested IDs absent from this dataset: {sorted(missing)[:5]}")
    reserved = {bid for batch in manifest["batches"].values() if not batch["results_ingested"]
                for bid in batch["expected"]}
    conflicts = requested & reserved if requested is not None else set()
    if conflicts:
        raise ValueError(f"requested IDs are reserved by pending batches: {sorted(conflicts)[:5]}")
    if requested is not None:
        eligible_ids = [bid for bid in dataset_ids if bid in requested]
    else:
        exact, carried = _snapshot_review_ids(state, digest, dataset_ids, helpers, dataset)
        reviewed = exact | carried
        eligible_ids = [bid for bid in dataset_ids if bid not in reviewed and bid not in reserved]
    selected_ids = eligible_ids[:limit]
    if not selected_ids:
        raise ValueError("no pending records available")
    selected = _load_records(dataset, selected_ids, digest, helpers)
    for row in selected:
        row["_review"].update({"batch_id": batch_id})
    state.mkdir(parents=True, exist_ok=True)
    out = state / f"{batch_id}.jsonl"
    content = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":"), default=str) + "\n" for r in selected)
    with out.open("x", encoding="utf-8", newline="\n") as f:
        f.write(content)
    packet_path = state / f"{batch_id}.compact.json"
    packet = _inspection_packet(selected, sha256(out), digest)
    with packet_path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(packet, f, ensure_ascii=False, separators=(",", ":"), default=str)
        f.write("\n")
    expected = {r[KEY]: r["_review"]["record_binding_sha256"] for r in selected}
    manifest["batches"][batch_id] = {"dataset_sha256": digest, "input_sha256": sha256(out), "count": len(selected),
                                     "compact_sha256": sha256(packet_path), "expected": expected, "results_ingested": False}
    atomic_json(state / "manifest.json", manifest)
    return out


def ingest(state: Path, batch_id: str, results_path: Path) -> int:
    manifest = read_manifest(state)
    batch = manifest["batches"].get(batch_id)
    if not batch:
        raise ValueError("unknown batch")
    if batch["results_ingested"]:
        raise ValueError("batch results are immutable and already ingested")
    input_path = state / f"{batch_id}.jsonl"
    if not input_path.is_file() or sha256(input_path) != batch["input_sha256"]:
        raise ValueError("batch input is missing or has changed since export")
    compact_path = state / f"{batch_id}.compact.json"
    if not compact_path.is_file() or sha256(compact_path) != batch["compact_sha256"]:
        raise ValueError("compact inspection packet is missing or has changed since export")
    compact = json.loads(compact_path.read_text(encoding="utf-8"))
    if compact.get("full_input_sha256") != batch["input_sha256"] or compact.get("dataset_sha256") != batch["dataset_sha256"]:
        raise ValueError("compact inspection packet bindings do not match the batch")
    results = validate_results(json.loads(results_path.read_text(encoding="utf-8")), batch["expected"])
    archive = state / f"{batch_id}_results.json"
    if archive.exists():
        raise ValueError("immutable result archive already exists")
    atomic_json(archive, results)
    input_rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    fingerprints = {str(r[KEY]): r["_review"]["content_fingerprint_sha256"] for r in input_rows}
    for r in results:
        manifest["completed"][str(r["building_id"])] = {
            "binding": r["record_binding_sha256"], "batch_id": batch_id,
            "dataset_sha256": batch["dataset_sha256"], "outcome": r["outcome"],
            "review_mode": r["review_mode"],
            "content_fingerprint_sha256": fingerprints[str(r["building_id"])],
            "sources_opened": sum(s.get("retrieval_status") == "opened_this_pass" for s in r["sources"])}
    batch["results_ingested"] = True
    batch["results_sha256"] = sha256(archive)
    atomic_json(state / "manifest.json", manifest)
    return len(results)


def import_legacy(state: Path, legacy: Path) -> int:
    manifest = read_manifest(state)
    if manifest.get("legacy_imported"):
        return 0
    count = 0
    for inp in sorted(legacy.glob("*.jsonl")):
        result_path = legacy / f"{inp.stem}_results.json"
        if not result_path.exists():
            continue
        input_rows = [json.loads(line) for line in inp.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = {str(r[KEY]): r["_review"]["record_binding_sha256"] for r in input_rows}
        # Earlier native batches did not record queries_used for every row. Keep
        # those reviews as legacy coverage, with the missing query log left explicit.
        results = validate_results(json.loads(result_path.read_text(encoding="utf-8")), expected, require_queries=False)
        for r in results:
            bid = str(r["building_id"])
            old = manifest["completed"].get(bid)
            if old and old["binding"] != r["record_binding_sha256"]:
                continue
            manifest["completed"][bid] = {"binding": r["record_binding_sha256"], "batch_id": "legacy:" + inp.stem,
                "dataset_sha256": input_rows[0]["_review"]["dataset_sha256"], "outcome": r["outcome"],
                "review_mode": "legacy_not_recorded",
                "sources_opened": sum(s.get("retrieval_status") == "opened_this_pass" for s in r["sources"])}
            count += 1
    manifest["legacy_imported"] = True
    manifest["legacy_source"] = str(legacy)
    atomic_json(state / "manifest.json", manifest)
    return count


def status(state: Path, dataset: Path | None = None) -> dict:
    manifest = read_manifest(state)
    distinct = len(manifest["completed"])
    modes = {}
    for record in manifest["completed"].values():
        mode = record.get("review_mode", "legacy_not_recorded")
        modes[mode] = modes.get(mode, 0) + 1
    opened = sum(record.get("sources_opened", 0) > 0 for record in manifest["completed"].values())
    current = None
    inventory = None
    if dataset:
        digest, ids, helpers = _load_dataset(dataset)
        exact, carried = _snapshot_review_ids(state, digest, ids, helpers, dataset)
        pending = len(ids) - len(exact) - len(carried)
        current = {"dataset_sha256": digest, "records": len(ids), "reviewed": len(exact),
                   "reviewed_exact_snapshot": len(exact), "carried_unchanged_rows": len(carried), "pending": pending}
        inventory = {"records": len(ids), "unique_ids_never_reviewed_across_snapshots": sum(bid not in manifest["completed"] for bid in ids)}
    return {"unique_record_ids_reviewed_across_snapshots": distinct, "review_modes": modes,
            "records_with_some_opened_sources": opened, "dataset_inventory": inventory,
            "current_snapshot_pending": current["pending"] if current else None, "current_dataset": current,
            "batches": {k: {"count": v["count"], "results_ingested": v["results_ingested"]} for k, v in manifest["batches"].items()},
            "fully_fact_verified_records": 0,
            "note": "Queued/exported rows do not count as reviewed; review coverage does not certify every field."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export"); p.add_argument("--dataset", type=Path, required=True); p.add_argument("--state-dir", type=Path, required=True)
    p.add_argument("--batch-id", required=True); p.add_argument("--limit", type=int, required=True); p.add_argument("--ids-file", type=Path)
    p = sub.add_parser("ingest"); p.add_argument("--state-dir", type=Path, required=True); p.add_argument("--batch-id", required=True); p.add_argument("--results", type=Path, required=True)
    p = sub.add_parser("status"); p.add_argument("--state-dir", type=Path, required=True); p.add_argument("--dataset", type=Path)
    p.add_argument("--legacy-dir", type=Path, help="Import exact completed native results before reporting status")
    p = sub.add_parser("import-legacy"); p.add_argument("--state-dir", type=Path, required=True); p.add_argument("--legacy-dir", type=Path, required=True)
    a = parser.parse_args(argv)
    if a.command == "export":
        print(export_batch(a.dataset, a.state_dir, a.batch_id, a.limit, a.ids_file))
    elif a.command == "ingest":
        print(json.dumps({"ingested": ingest(a.state_dir, a.batch_id, a.results)}))
    elif a.command == "status":
        if a.legacy_dir:
            import_legacy(a.state_dir, a.legacy_dir)
        print(json.dumps(status(a.state_dir, a.dataset), ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"legacy_rows_imported": import_legacy(a.state_dir, a.legacy_dir)}))


if __name__ == "__main__":
    main()
