"""Rebuild V2 classifications with the frozen original method.

The replay consumes a checksummed input manifest and runs this fixed sequence:

1. full original-rule rerun after corrected physical and BDBIAR scope inputs;
2. original LLM checkpoints only when the complete prompt is unchanged;
3. fail-closed VLM checkpoint handling, followed by verified nonbuilding
   physical-scope exclusions;
4. original nine-feature ML training and calibration;
5. original Step 10 public classification standardization.

The default path is offline.  It never reads web-derived function evidence and
never calls a text or vision service.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = PROJECT_ROOT / "src" / "frozen_original_method"
FROZEN_MANIFEST = FROZEN_ROOT / "source_manifest.json"
STAGES = {
    3: {
        "script": PROJECT_ROOT / "src" / "replay_original_v2_stage3.py",
        "outputs": [
            "checkpoints/step3_rule_engine_classification.csv",
            "checkpoints/stage3_foundation_scope_comparison.csv",
            "checkpoints/stage3_build_manifest.json",
        ],
    },
    5: {
        "script": PROJECT_ROOT / "src" / "replay_original_v2_stage5.py",
        "outputs": [
            "checkpoints/step4_exact_input_llm_evidence.csv",
            "checkpoints/step5_merged_classification.csv",
            "checkpoints/stage4_invalidated_llm_cache.csv",
            "checkpoints/stage5_classification_changes.csv",
            "checkpoints/stage5_build_manifest.json",
        ],
    },
    7: {
        "script": PROJECT_ROOT / "src" / "replay_original_v2_stage7.py",
        "outputs": [
            "checkpoints/step7_corrected.csv",
            "checkpoints/stage7_classification_changes.csv",
            "checkpoints/stage7_withheld_vlm_decisions.csv",
            "checkpoints/stage7_build_manifest.json",
        ],
    },
    9: {
        "script": PROJECT_ROOT / "src" / "replay_original_v2_stage89.py",
        "arguments": ["--run"],
        "outputs": [
            "ml/features_base_indexed.pkl",
            "ml/features_X_all.pkl",
            "ml/features_X_train_ml.pkl",
            "ml/features_y_multilabel.pkl",
            "ml/floor_missing_rule_mixture_audit.csv",
            "ml/floor_missing_rule_mixture_summary.json",
            "ml/height_missing_ml_sensitivity_audit.csv",
            "ml/height_missing_ml_sensitivity_summary.json",
            "ml/models.pkl",
            "ml/probabilities.csv",
            "ml/step8_feature_base_calibrated.geojson",
            "ml/step9_corrected_ml_calibrated.csv",
            "ml/step9_corrected_ml_calibrated.geojson",
            "ml/training_samples.pkl",
            "ml/metrics.json",
            "ml/hash_manifest.json",
        ],
    },
    10: {
        "script": PROJECT_ROOT / "src" / "replay_original_v2_stage10.py",
        "outputs": [
            "ml/step10_original_method_standardized.csv",
            "ml/step10_original_method_standardization_changes.csv",
            "ml/step10_original_method_manifest.json",
        ],
    },
}

REQUIRED_INPUT_ROLES = {
    "legacy_stage2",
    "legacy_stage3",
    "legacy_stage4",
    "legacy_stage5",
    "verified_physical",
    "bdbiar_scope",
    "nonbuilding_scope",
    "gfa_rule_delta",
    "llm_review_summary",
    "llm_prompt_contract",
    "native_reviews",
    "stage6_suggestions",
    "vlm_audit",
}

# These files define the accepted release lineage.  The remaining inputs are
# also hash-checked against the supplied manifest, but their release-specific
# digests can change only by replacing that manifest deliberately.
PINNED_INPUT_HASHES = {
    "legacy_stage2": "f37a03a0d076e97d77d2c700b0153fdb0be3415d70a81e100145761853b2885a",
    "legacy_stage3": "c8b36fa65e5cf07fca77ce4cddf4a46a1e7902fdb82b369bc8e960fa777ee010",
    "legacy_stage5": "4adbab285f0acd50f258a7b2c1a5fad01867bec35ea40050e6ec17f4d85b88fa",
    "verified_physical": "783e30db5240eb02cf6217e03707f98395a129d942968ff6115cc1b23b418c88",
    "bdbiar_scope": "389fb63fc8c59ed638fdda5e1a443058af233734ee8bcba719ae61a9fba1886a",
    "nonbuilding_scope": "2c14d4adbc75edaf79a621f9b6228b37c0bf90b9554ac06b0cab771ea24f1e38",
    "stage6_suggestions": "09898410ac58f67f9932055ad0339cbda4078d5c5e301de02dcc4d9c5f947dba",
}

ENV_BY_ROLE = {
    "legacy_stage2": "HKBFETD_REPLAY_LEGACY_STAGE2",
    "legacy_stage3": "HKBFETD_REPLAY_LEGACY_STAGE3",
    "legacy_stage4": "HKBFETD_REPLAY_LEGACY_STAGE4",
    "legacy_stage5": "HKBFETD_REPLAY_LEGACY_STAGE5",
    "verified_physical": "HKBFETD_REPLAY_VERIFIED_PHYSICAL",
    "bdbiar_scope": "HKBFETD_REPLAY_BDBIAR_SCOPE",
    "nonbuilding_scope": "HKBFETD_REPLAY_NONBUILDING_SCOPE",
    "gfa_rule_delta": "HKBFETD_REPLAY_GFA_DELTA",
    "llm_review_summary": "HKBFETD_REPLAY_LLM_REVIEW_SUMMARY",
    "llm_prompt_contract": "HKBFETD_REPLAY_LLM_PROMPT_CONTRACT",
    "native_reviews": "HKBFETD_REPLAY_NATIVE_REVIEWS",
    "stage6_suggestions": "HKBFETD_REPLAY_STAGE6_SUGGESTIONS",
    "vlm_audit": "HKBFETD_REPLAY_VLM_AUDIT",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_frozen_method() -> dict:
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    observed = {}
    for relative, expected in manifest["files"].items():
        path = FROZEN_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Frozen original-method source missing: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Frozen original-method source changed: {relative}; "
                f"expected {expected}, got {actual}"
            )
        observed[relative] = actual
    return {
        "source_git_commit": manifest["source_git_commit"],
        "classification_order": manifest["classification_order"],
        "files": observed,
    }


def resolve_input_manifest(path: Path) -> tuple[dict, dict[str, Path]]:
    manifest_path = path.resolve()
    document = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if document.get("schema_version") != 1 or not isinstance(document.get("files"), dict):
        raise ValueError("Replay input manifest must have schema_version=1 and a files object")
    missing_roles = REQUIRED_INPUT_ROLES - set(document["files"])
    extra_roles = set(document["files"]) - REQUIRED_INPUT_ROLES
    if missing_roles or extra_roles:
        raise ValueError(
            f"Replay input roles differ; missing={sorted(missing_roles)}, "
            f"unexpected={sorted(extra_roles)}"
        )

    base = manifest_path.parent
    resolved = {}
    for role in sorted(REQUIRED_INPUT_ROLES):
        record = document["files"][role]
        relative = Path(record["path"])
        if relative.is_absolute():
            raise ValueError(f"Input path for {role} must be relative to the manifest")
        candidate = (base / relative).resolve()
        try:
            candidate.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"Input path for {role} escapes the manifest directory") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"Replay input missing for {role}: {candidate}")
        expected = str(record["sha256"]).lower()
        actual = sha256(candidate)
        if actual != expected:
            raise RuntimeError(
                f"Replay input changed for {role}; expected {expected}, got {actual}"
            )
        pinned = PINNED_INPUT_HASHES.get(role)
        if pinned is not None and actual != pinned:
            raise RuntimeError(
                f"Replay lineage mismatch for {role}; expected release pin {pinned}, got {actual}"
            )
        if "building_function_evidence" in candidate.name.lower():
            raise ValueError("Web-derived building function evidence is not a replay input")
        resolved[role] = candidate
    return document, resolved


def stage_signature(stage: int, input_document: dict, frozen: dict, upstream: str) -> str:
    script = STAGES[stage]["script"]
    return stable_hash(
        {
            "stage": stage,
            "script_sha256": sha256(script),
            "input_manifest": input_document,
            "frozen_method": frozen,
            "upstream": upstream,
        }
    )


def output_records(output_root: Path, names: list[str]) -> dict:
    records = {}
    for name in names:
        path = output_root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Replay stage did not create required output: {path}")
        records[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return records


def can_resume(record: dict, signature: str, output_root: Path, outputs: list[str]) -> bool:
    if record.get("signature") != signature:
        return False
    expected = record.get("outputs", {})
    for name in outputs:
        path = output_root / name
        if not path.is_file() or expected.get(name, {}).get("sha256") != sha256(path):
            return False
    return True


def run(args: argparse.Namespace) -> None:
    frozen = verify_frozen_method()
    input_document, inputs = resolve_input_manifest(args.inputs_manifest)
    check = {
        "status": "inputs_verified",
        "inputs_manifest": str(args.inputs_manifest.resolve()),
        "input_roles": {
            role: {"path": str(path), "sha256": sha256(path)}
            for role, path in sorted(inputs.items())
        },
        "frozen_method": frozen,
        "external_ai_requests": "disabled",
        "web_function_evidence": "excluded",
    }
    if args.check_only:
        print(json.dumps(check, ensure_ascii=False, indent=2))
        return

    output_root = args.output_dir.resolve()
    checkpoints = output_root / "checkpoints"
    ml_output = output_root / "ml"
    logs = output_root / "logs"
    for directory in (checkpoints, ml_output, logs):
        directory.mkdir(parents=True, exist_ok=True)

    run_manifest_path = output_root / "original_method_replay_manifest.json"
    previous = {}
    if args.resume and run_manifest_path.is_file():
        previous = json.loads(run_manifest_path.read_text(encoding="utf-8"))

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "HKBFETD_REPLAY_FROZEN_METHOD": str(FROZEN_ROOT),
            "HKBFETD_REPLAY_OUTPUT_DIR": str(checkpoints),
            "HKBFETD_REPLAY_ML_OUTPUT_DIR": str(ml_output),
        }
    )
    for role, variable in ENV_BY_ROLE.items():
        environment[variable] = str(inputs[role])

    run_manifest = {
        "schema_version": 1,
        "dataset_version": "2.0-corrected",
        "method": "full frozen original rule rerun -> exact-prompt LLM checkpoints -> fail-closed VLM checkpoints -> nine-feature ML -> original Step 10",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        **check,
        "stages": {},
    }
    upstream = stable_hash({"inputs": input_document, "frozen": frozen})
    for stage in (3, 5, 7, 9, 10):
        if stage > args.through:
            break
        spec = STAGES[stage]
        signature = stage_signature(stage, input_document, frozen, upstream)
        old_record = previous.get("stages", {}).get(str(stage), {})
        if args.resume and can_resume(old_record, signature, output_root, spec["outputs"]):
            record = old_record
            print(f"[INFO] Original-method stage {stage}: verified checkpoint reused", flush=True)
        else:
            command = [sys.executable, "-u", str(spec["script"]), *spec.get("arguments", [])]
            log_path = logs / f"stage_{stage}.txt"
            print(f"[INFO] Original-method stage {stage}: running", flush=True)
            with log_path.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode:
                raise RuntimeError(
                    f"Original-method stage {stage} failed with code "
                    f"{completed.returncode}; see {log_path}"
                )
            record = {
                "signature": signature,
                "script": str(spec["script"].relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "script_sha256": sha256(spec["script"]),
                "outputs": output_records(output_root, spec["outputs"]),
                "log": str(log_path.relative_to(output_root)).replace("\\", "/"),
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            }
        run_manifest["stages"][str(stage)] = record
        upstream = stable_hash(record)
        run_manifest_path.write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    run_manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    run_manifest["final_classification"] = (
        "ml/step10_original_method_standardized.csv" if args.through == 10 else None
    )
    run_manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[INFO] Original-method replay complete through stage {args.through}: {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/v2_original_replay"))
    parser.add_argument("--through", type=int, choices=(3, 5, 7, 9, 10), default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
