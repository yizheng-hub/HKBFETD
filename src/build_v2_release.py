from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

try:
    import geopandas as gpd
except ModuleNotFoundError:  # Unit tests may exercise ordinary Parquet only.
    gpd = None

try:
    import pyogrio
except ModuleNotFoundError:
    pyogrio = None


ID_COLUMN = "Building_ID_建筑ID"
CSV_NAME = "HKBFETD_v2.csv"
PARQUET_NAME = "HKBFETD_v2.parquet"
GEOJSON_NAME = "HKBFETD_v2.geojson"
MANIFEST_NAME = "release_manifest.json"
CHECKSUMS_NAME = "SHA256SUMS.txt"
DEFAULT_ARCHIVE_NAME = "HKBFETD_v2_release.zip"


@dataclass(frozen=True)
class ReleaseInputs:
    csv_path: Path
    parquet_path: Path
    geojson_path: Path
    schema_path: Path
    readme_path: Path
    model_paths: tuple[Path, ...]
    metrics_path: Path
    validation_report_paths: tuple[Path, ...]
    reproduction_inputs_path: Path
    source_snapshot_path: Path
    source_evidence_paths: tuple[Path, ...] = ()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _all_sources(inputs: ReleaseInputs) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = [
        ("csv", inputs.csv_path),
        ("parquet", inputs.parquet_path),
        ("geojson", inputs.geojson_path),
        ("schema", inputs.schema_path),
        ("readme", inputs.readme_path),
        ("metrics", inputs.metrics_path),
        ("reproduction_inputs", inputs.reproduction_inputs_path),
        ("source_snapshot", inputs.source_snapshot_path),
    ]
    sources.extend(("model", path) for path in inputs.model_paths)
    sources.extend(
        ("validation_report", path) for path in inputs.validation_report_paths
    )
    sources.extend(("source_evidence", path) for path in inputs.source_evidence_paths)
    return [(role, _as_path(path)) for role, path in sources]


def _destination_map(inputs: ReleaseInputs) -> list[tuple[str, Path, PurePosixPath]]:
    result: list[tuple[str, Path, PurePosixPath]] = [
        ("geojson", _as_path(inputs.geojson_path), PurePosixPath(GEOJSON_NAME)),
        ("schema", _as_path(inputs.schema_path), PurePosixPath("schema.json")),
        ("readme", _as_path(inputs.readme_path), PurePosixPath("README_zh.md")),
        (
            "metrics",
            _as_path(inputs.metrics_path),
            PurePosixPath("models") / inputs.metrics_path.name,
        ),
        (
            "reproduction_inputs",
            _as_path(inputs.reproduction_inputs_path),
            PurePosixPath("reproduction") / inputs.reproduction_inputs_path.name,
        ),
        (
            "source_snapshot",
            _as_path(inputs.source_snapshot_path),
            PurePosixPath("reproduction") / inputs.source_snapshot_path.name,
        ),
    ]
    result.extend(
        ("model", _as_path(path), PurePosixPath("models") / path.name)
        for path in inputs.model_paths
    )
    result.extend(
        (
            "validation_report",
            _as_path(path),
            PurePosixPath("validation") / path.name,
        )
        for path in inputs.validation_report_paths
    )
    result.extend(
        (
            "source_evidence",
            _as_path(path),
            PurePosixPath("evidence") / path.name,
        )
        for path in inputs.source_evidence_paths
    )
    destinations = [str(item[2]).casefold() for item in result]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Two explicitly supplied inputs map to the same package path")
    return result


def validate_input_files(inputs: ReleaseInputs, out: Path) -> None:
    if not inputs.model_paths:
        raise ValueError("At least one --model file is required")
    if not inputs.validation_report_paths:
        raise ValueError("At least one --validation-report file is required")
    resolved_out = _as_path(out)
    for role, path in _all_sources(inputs):
        if not path.exists():
            raise FileNotFoundError(f"Missing required {role} input: {path}")
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"The {role} input must be an explicit regular file, not a directory "
                f"or symlink: {path}"
            )
        if path == resolved_out or resolved_out in path.parents:
            raise ValueError(f"Input cannot be inside the output directory: {path}")
    _destination_map(inputs)


def _blank_to_na(series: pd.Series) -> tuple[pd.Series, int]:
    if not (isinstance(series.dtype, pd.StringDtype) or series.dtype == object):
        return series, 0
    blank = series.map(lambda value: isinstance(value, str) and not value.strip())
    count = int(blank.sum())
    if count:
        series = series.mask(blank, pd.NA)
    return series, count


def _read_parquet(path: Path) -> pd.DataFrame:
    metadata = pq.read_schema(path).metadata or {}
    if b"geo" in metadata:
        if gpd is None:
            raise RuntimeError(
                "GeoPandas is required to normalise the final GeoParquet without "
                "discarding its CRS/geometry metadata"
            )
        return gpd.read_parquet(path)
    if gpd is None:
        return pd.read_parquet(path)
    try:
        return gpd.read_parquet(path)
    except ValueError as exc:
        if "Missing geo metadata" not in str(exc):
            raise
        return pd.read_parquet(path)


def _normalise_parquet(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    result = frame.copy()
    replacements: dict[str, int] = {}
    geometry_name = (
        result.geometry.name
        if gpd is not None and isinstance(result, gpd.GeoDataFrame)
        else None
    )
    for column in result.columns:
        if column == geometry_name:
            continue
        result[column], count = _blank_to_na(result[column])
        if count:
            replacements[column] = count
        if is_numeric_dtype(result[column].dtype):
            values = result[column].dropna().to_numpy()
            if values.size and not np.isfinite(values).all():
                raise ValueError(f"Non-finite numeric value in Parquet column {column}")
    return result, replacements


def _normalise_csv_strings(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        result[column], _ = _blank_to_na(result[column])
    return result


def _normalise_ids(series: pd.Series, label: str) -> pd.Series:
    ids = series.astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any():
        raise ValueError(f"{label} contains an empty {ID_COLUMN}")
    if ids.duplicated().any():
        examples = ids.loc[ids.duplicated(keep=False)].head(5).tolist()
        raise ValueError(f"{label} contains duplicate {ID_COLUMN} values: {examples}")
    return ids


def _compare_column(csv_series: pd.Series, parquet_series: pd.Series, name: str) -> None:
    csv_null = csv_series.isna()
    parquet_null = parquet_series.isna()
    if not csv_null.equals(parquet_null):
        count = int((csv_null != parquet_null).sum())
        raise ValueError(f"CSV/Parquet null masks differ in {name}: {count} rows")
    present = ~parquet_null
    if not present.any():
        return
    left = csv_series.loc[present]
    right = parquet_series.loc[present]
    if is_numeric_dtype(parquet_series.dtype) and not is_bool_dtype(parquet_series.dtype):
        try:
            left_numeric = pd.to_numeric(left, errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CSV has non-numeric data in numeric column {name}") from exc
        right_numeric = pd.to_numeric(right, errors="raise").to_numpy(dtype=float)
        equal = np.isclose(left_numeric, right_numeric, rtol=1e-12, atol=1e-9)
    elif is_bool_dtype(parquet_series.dtype):
        mapping = {"true": True, "false": False, "1": True, "0": False}
        lowered = left.astype("string").str.strip().str.casefold()
        if not lowered.isin(mapping).all():
            raise ValueError(f"CSV has non-boolean data in boolean column {name}")
        equal = lowered.map(mapping).to_numpy() == right.astype(bool).to_numpy()
    elif is_datetime64_any_dtype(parquet_series.dtype):
        equal = pd.to_datetime(left, errors="raise").to_numpy() == right.to_numpy()
    else:
        equal = left.astype("string").to_numpy() == right.astype("string").to_numpy()
    if not bool(np.asarray(equal).all()):
        count = int((~np.asarray(equal)).sum())
        raise ValueError(f"CSV/Parquet values differ in {name}: {count} rows")


def validate_and_normalise_tables(
    csv_path: Path, parquet_path: Path
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    parquet = _read_parquet(parquet_path)
    parquet, blank_replacements = _normalise_parquet(parquet)
    geometry_name = (
        parquet.geometry.name
        if gpd is not None and isinstance(parquet, gpd.GeoDataFrame)
        else None
    )
    parquet_attributes = [column for column in parquet.columns if column != geometry_name]
    csv_frame = pd.read_csv(
        csv_path,
        dtype="string",
        keep_default_na=False,
        low_memory=False,
        encoding="utf-8-sig",
    )
    csv_frame = _normalise_csv_strings(csv_frame)
    if csv_frame.columns.tolist() != parquet_attributes:
        missing = sorted(set(parquet_attributes) - set(csv_frame.columns))
        extra = sorted(set(csv_frame.columns) - set(parquet_attributes))
        order_only = not missing and not extra
        raise ValueError(
            "CSV columns must exactly match Parquet non-geometry columns and order; "
            f"missing={missing}, extra={extra}, order_only_mismatch={order_only}"
        )
    if ID_COLUMN not in csv_frame.columns:
        raise ValueError(f"Both tables must contain {ID_COLUMN}")
    csv_ids = _normalise_ids(csv_frame[ID_COLUMN], "CSV")
    parquet_ids = _normalise_ids(parquet[ID_COLUMN], "Parquet")
    csv_frame[ID_COLUMN] = csv_ids
    parquet[ID_COLUMN] = parquet_ids
    if set(csv_ids) != set(parquet_ids):
        raise ValueError("CSV and Parquet contain different Building ID sets")
    csv_aligned = csv_frame.set_index(ID_COLUMN).loc[parquet_ids].reset_index()
    parquet_aligned = parquet.reset_index(drop=True)
    for column in parquet_attributes:
        _compare_column(csv_aligned[column], parquet_aligned[column], column)
    return parquet_aligned, parquet_attributes, {
        "rows": int(len(parquet_aligned)),
        "columns": len(parquet_attributes),
        "id_column": ID_COLUMN,
        "blank_strings_normalised_to_null": blank_replacements,
        "csv_parquet_semantic_match": True,
    }


def _load_json_file(path: Path, role: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {role}: {path}") from exc


def _validate_json_file(path: Path, role: str) -> None:
    _load_json_file(path, role)


def _validate_validation_report(path: Path) -> None:
    document = _load_json_file(path, "validation report")
    passed = False

    def visit(value: Any) -> None:
        nonlocal passed
        if isinstance(value, dict):
            for key, item in value.items():
                key_text = str(key).casefold()
                is_outcome = key_text in {
                    "status",
                    "verdict",
                    "result",
                    "conclusion",
                } or key_text.endswith(("_status", "_verdict", "_result", "_conclusion"))
                is_pass_flag = key_text in {"pass", "passed"} or key_text.endswith(
                    ("_pass", "_passed")
                )
                if is_outcome and isinstance(item, bool):
                    if not item:
                        raise ValueError(f"Validation report has a failing {key}: {path}")
                    passed = True
                elif is_outcome and isinstance(item, str):
                    tokens = set(re.findall(r"[a-z]+", item.casefold()))
                    if tokens & {"fail", "failed", "failure", "blocked", "hold", "error", "errors"}:
                        raise ValueError(
                            f"Validation report has a blocked/failing {key}={item!r}: {path}"
                        )
                    pass_tokens = tokens & {
                        "pass",
                        "passed",
                        "passing",
                        "success",
                        "successful",
                        "succeeded",
                        "ok",
                        "accepted",
                        "approved",
                    }
                    if pass_tokens and "not" in tokens:
                        raise ValueError(
                            f"Validation report does not pass: {key}={item!r}: {path}"
                        )
                    if pass_tokens:
                        passed = True
                if is_pass_flag and isinstance(item, bool):
                    if not item:
                        raise ValueError(f"Validation report has {key}=false: {path}")
                    passed = True
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(document)
    if not passed:
        raise ValueError(f"Validation report has no explicit passing conclusion: {path}")


def _validate_geojson(
    path: Path,
    attributes: list[str],
    expected_ids: pd.Series,
) -> dict[str, Any]:
    if pyogrio is None:
        raise RuntimeError("pyogrio is required for release GeoJSON validation")
    size = path.stat().st_size
    if size < 32:
        raise ValueError(f"GeoJSON input is unexpectedly small: {path}")
    with path.open("rb") as handle:
        head = handle.read(min(size, 65536))
        handle.seek(max(0, size - 65536))
        tail = handle.read()
    if b"FeatureCollection" not in head or not tail.rstrip().endswith(b"}"):
        raise ValueError(f"GeoJSON does not have a valid FeatureCollection envelope: {path}")
    try:
        info = pyogrio.read_info(path, force_feature_count=True)
    except Exception as exc:
        raise ValueError(f"GeoJSON cannot be read by pyogrio: {path}") from exc
    if info.get("driver") != "GeoJSON":
        raise ValueError(f"GeoJSON input is not a GeoJSON FeatureCollection: {path}")
    feature_count = int(info.get("features", -1))
    if feature_count != len(expected_ids):
        raise ValueError(
            f"GeoJSON feature count differs from CSV/Parquet: "
            f"{feature_count} != {len(expected_ids)}"
        )
    fields = [str(value) for value in info.get("fields", [])]
    if fields != attributes:
        missing = sorted(set(attributes) - set(fields))
        extra = sorted(set(fields) - set(attributes))
        order_only = not missing and not extra
        raise ValueError(
            "GeoJSON property fields must exactly match CSV/Parquet columns and order; "
            f"missing={missing}, extra={extra}, order_only_mismatch={order_only}"
        )
    if str(info.get("crs", "")).upper() != "EPSG:4326":
        raise ValueError(
            f"GeoJSON CRS must be RFC 7946 WGS84 (EPSG:4326), got {info.get('crs')!r}"
        )
    try:
        id_frame = pyogrio.read_dataframe(
            path,
            columns=[ID_COLUMN],
            read_geometry=False,
        )
    except Exception as exc:
        raise ValueError(f"GeoJSON ID column cannot be read: {path}") from exc
    if id_frame.columns.tolist() != [ID_COLUMN]:
        raise ValueError(f"GeoJSON does not expose the required {ID_COLUMN} property")
    geojson_ids = _normalise_ids(id_frame[ID_COLUMN], "GeoJSON")
    expected = _normalise_ids(expected_ids, "CSV/Parquet")
    if set(geojson_ids) != set(expected):
        raise ValueError("GeoJSON and CSV/Parquet contain different Building ID sets")
    return {
        "feature_collection": True,
        "features": feature_count,
        "property_fields_match": True,
        "unique_id_set_matches": True,
        "crs": "EPSG:4326",
    }


def validate_release_inputs(
    inputs: ReleaseInputs, out: Path
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    validate_input_files(inputs, out)
    _validate_json_file(_as_path(inputs.schema_path), "schema")
    for report in inputs.validation_report_paths:
        if report.suffix.casefold() == ".json":
            _validate_validation_report(_as_path(report))
    frame, attributes, validation = validate_and_normalise_tables(
        _as_path(inputs.csv_path), _as_path(inputs.parquet_path)
    )
    validation["geojson"] = _validate_geojson(
        _as_path(inputs.geojson_path),
        attributes,
        frame[ID_COLUMN],
    )
    return frame, attributes, validation


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _write_normalised_tables(
    frame: pd.DataFrame, attributes: list[str], destination: Path
) -> None:
    frame[attributes].to_csv(
        destination / CSV_NAME,
        index=False,
        encoding="utf-8-sig",
        na_rep="",
        quoting=csv.QUOTE_MINIMAL,
    )
    frame.to_parquet(destination / PARQUET_NAME, index=False)


def _payload_files(root: Path) -> list[Path]:
    excluded = {MANIFEST_NAME, CHECKSUMS_NAME}
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.relative_to(root).as_posix() not in excluded
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _write_zip(root: Path, archive_path: Path, members: Iterable[Path]) -> None:
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for path in members:
            archive.write(path, arcname=path.relative_to(root).as_posix())


def build_release(
    out: Path,
    inputs: ReleaseInputs,
    *,
    release_version: str = "2.0",
    archive_name: str = DEFAULT_ARCHIVE_NAME,
    dry_run: bool = False,
) -> dict[str, Any]:
    out = _as_path(out)
    if out.exists():
        raise FileExistsError(f"Output directory already exists; refusing to overwrite: {out}")
    if Path(archive_name).name != archive_name or not archive_name.casefold().endswith(".zip"):
        raise ValueError("--archive-name must be a plain .zip filename")
    frame, attributes, table_validation = validate_release_inputs(inputs, out)
    provenance = [
        {
            "role": role,
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for role, path in _all_sources(inputs)
    ]
    plan = {
        "status": "dry_run_passed" if dry_run else "passed",
        "release_version": release_version,
        "output_directory": str(out),
        "archive_name": archive_name,
        "table_validation": table_validation,
        "explicit_input_count": len(provenance),
    }
    if dry_run:
        return plan
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}.building-", dir=out.parent) as temp:
        staging = Path(temp) / out.name
        staging.mkdir()
        _write_normalised_tables(frame, attributes, staging)
        for _, source, relative in _destination_map(inputs):
            _copy_file(source, staging / Path(relative))
        payload = _payload_files(staging)
        manifest = {
            "dataset": "HKBFETD",
            "release_version": release_version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "table_validation": table_validation,
            "serialization": {
                "csv": "UTF-8-SIG; blank fields are null",
                "parquet": "blank strings normalised to null; source geometry retained",
                "geojson": "copied byte-for-byte from the explicitly supplied final GeoJSON",
            },
            "input_provenance": provenance,
            "package_files": {
                path.relative_to(staging).as_posix(): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in payload
            },
            "packaging_policy": {
                "implicit_directory_scans": False,
                "symlink_inputs_allowed": False,
                "only_explicit_files_included": True,
            },
        }
        manifest_path = staging / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        checksummed = [*payload, manifest_path]
        checksums_path = staging / CHECKSUMS_NAME
        checksums_path.write_text(
            "".join(
                f"{sha256(path)}  {path.relative_to(staging).as_posix()}\n"
                for path in checksummed
            ),
            encoding="utf-8",
        )
        archive_path = staging / archive_name
        _write_zip(staging, archive_path, [*checksummed, checksums_path])
        archive_digest_path = staging / f"{archive_name}.sha256"
        archive_digest_path.write_text(
            f"{sha256(archive_path)}  {archive_name}\n", encoding="utf-8"
        )
        for attempt in range(6):
            try:
                staging.replace(out)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(2**attempt)
    plan["outputs"] = sorted(
        path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()
    )
    plan["archive_sha256"] = sha256(out / archive_name)
    return plan


def _paths(values: list[Path] | None) -> tuple[Path, ...]:
    return tuple(values or ())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build one HKBFETD V2.0 Zenodo release folder from explicitly supplied, "
            "already-final artifacts. No work directory is scanned implicitly."
        )
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--geojson", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument(
        "--validation-report", type=Path, action="append", required=True
    )
    parser.add_argument("--reproduction-inputs", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument(
        "--source-evidence",
        type=Path,
        action="append",
        help="Optional, explicit source-evidence file; repeat for multiple files",
    )
    parser.add_argument("--release-version", default="2.0")
    parser.add_argument("--archive-name", default=DEFAULT_ARCHIVE_NAME)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all inputs and CSV/Parquet semantics without writing output",
    )
    args = parser.parse_args()
    inputs = ReleaseInputs(
        csv_path=args.csv,
        parquet_path=args.parquet,
        geojson_path=args.geojson,
        schema_path=args.schema,
        readme_path=args.readme,
        model_paths=_paths(args.model),
        metrics_path=args.metrics,
        validation_report_paths=_paths(args.validation_report),
        reproduction_inputs_path=args.reproduction_inputs,
        source_snapshot_path=args.source_snapshot,
        source_evidence_paths=_paths(args.source_evidence),
    )
    result = build_release(
        args.out,
        inputs,
        release_version=args.release_version,
        archive_name=args.archive_name,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
