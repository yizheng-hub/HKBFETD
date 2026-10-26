"""Assemble the public HKBFETD V2.0 tables from two authoritative inputs.

The classification CSV supplies the frozen original-method classification and
the public source-description fields.  The verified physical GeoParquet is the
sole source for geometry and adopted footprint, floor and height values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping
from shapely.geometry.polygon import orient

import release_classification_integrity as classification_integrity


ID = "Building_ID_建筑ID"
RAW_ID = "BUILDINGSTRUCTUREID"
CSUID = "BUILDINGCSUID"
GEOMETRY = "geometry"

PUBLIC_FIELDS: tuple[tuple[str, str], ...] = (
    (ID, RAW_ID),
    ("Footprint_Area_sqm_占地面积", "Footprint_Area_sqm_占地面积"),
    ("Floors_Count_地上层数", "Floors_Count_地上层数"),
    ("Height_m_建筑高度", "Height_m_建筑高度"),
    ("UBEM_Main_Class_主类别", "Calibrated_Main_Class"),
    ("UBEM_Sub_Class_子类别", "Calibrated_Sub_Class"),
    ("UBEM_Mixed_Proportions_混合比例", "Calibrated_Mix_Proportion"),
    ("Classification_Source_分类来源", "Calibrated_Source"),
    ("Classification_Notes_分类备注", "Calibrated_Notes"),
    ("ML_Prediction_Confidence_机器学习置信度", "ML_Confidence"),
    ("ML_Prob_Residential_住宅概率", "proba_住宅类别"),
    ("ML_Prob_Commercial_商业概率", "proba_商业类别"),
    ("ML_Prob_Industrial_工业概率", "proba_工业类别"),
    ("ML_Height_Sensitivity_Status_高度缺失敏感性状态", "ML_Height_Sensitivity_Status"),
    (
        "Floor_Missing_Classification_Gate_层数缺失分类门",
        "Floor_Missing_Classification_Gate",
    ),
    ("LandsD_Structure_Type_地政署建筑类型", "BUILDINGSTRUCTURETYPE"),
    ("LandsD_Category_地政署分类", "CATEGORY"),
    ("LandsD_Status_地政署状态", "STATUS"),
    ("LandsD_Registered_GFA_地政署注册面积", "GROSSFLOORAREA"),
    ("LandsD_Building_Name_EN_地政署英文名", "BUILDINGNAMEEN"),
    ("LandsD_Building_Name_TC_地政署中文名", "BUILDINGNAMETC"),
    ("BDBIAR_Record_ID_屋宇署记录ID", "BDBIAR_OBJECTID"),
    ("BDBIAR_Classification_屋宇署分类", "BDBIAR_CLASS"),
    ("BDBIAR_Building_Age_Year_屋宇署楼龄年份", "BDBIAR_AGE"),
    ("BDBIAR_Address_TC_屋宇署中文地址", "BDBIAR_ADDRESS_C"),
    ("Dist_to_BDBIAR_m_距屋宇署点位距离", "distance_to_bdbiar"),
    ("BDBIAR_Match_Quality_屋宇署匹配质量", "BDBIAR_Match_Quality"),
    ("BDBIAR_Source_Attributes_Usable_屋宇署属性可采用", "BDBIAR_Source_Attributes_Usable"),
    ("BDBIAR_Source_Integrity_Status_屋宇署来源完整性", "BDBIAR_Source_Integrity_Status"),
    ("BDBIAR_Candidate_Record_ID_屋宇署候选记录ID", "BDBIAR_OBJECTID_CANDIDATE"),
    ("BDBIAR_Candidate_Classification_屋宇署候选分类", "BDBIAR_CLASS_CANDIDATE"),
    ("BDBIAR_Candidate_Building_Age_Year_屋宇署候选楼龄年份", "BDBIAR_AGE_CANDIDATE"),
    ("BDBIAR_Candidate_Address_TC_屋宇署候选中文地址", "BDBIAR_ADDRESS_C_CANDIDATE"),
    ("Crowdsourced_POIs_Matched_众包POI匹配详情", "Matched_POIs"),
    ("OSM_Record_ID_OSM记录ID", "osmid"),
    ("OSM_Building_Tag_OSM建筑标签", "building"),
    ("OSM_Amenity_Tag_OSM设施标签", "amenity"),
    ("OSM_Shop_Tag_OSM商铺标签", "shop"),
    ("OZP_Zone_Code_法定图则代码", "OZP_ZONE_LABEL"),
    ("OZP_Zone_Desc_EN_法定图则英文描述", "OZP_DESC_ENG"),
)

PHYSICAL_PUBLIC_FIELDS = {
    ID,
    "Footprint_Area_sqm_占地面积",
    "Floors_Count_地上层数",
    "Height_m_建筑高度",
    "LandsD_Structure_Type_地政署建筑类型",
    "LandsD_Category_地政署分类",
    "LandsD_Status_地政署状态",
    "LandsD_Registered_GFA_地政署注册面积",
}

STRUCTURE_SCOPE_FIELDS = (
    "Structure_Use_Type",
    "Structure_Use_Name",
    "Structure_Use_Source",
    "Structure_Use_Evidence",
    "Structure_Use_Quality",
    "Energy_Assessment_Scope",
)

PHYSICAL_CORE = (
    "Footprint_Area_sqm_占地面积",
    "Floors_Count_地上层数",
    "Height_m_建筑高度",
)

CLASSIFICATION_STRING_FIELDS = {
    RAW_ID,
    CSUID,
    "Calibrated_Main_Class",
    "Calibrated_Sub_Class",
    "Calibrated_Mix_Proportion",
    "Calibrated_Source",
    "Calibrated_Notes",
    "ML_Height_Sensitivity_Status",
    "Floor_Missing_Classification_Gate",
    "BUILDINGSTRUCTURETYPE",
    "STATUS",
    "BUILDINGNAMEEN",
    "BUILDINGNAMETC",
    "BDBIAR_OBJECTID",
    "BDBIAR_CLASS",
    "BDBIAR_ADDRESS_C",
    "BDBIAR_Match_Quality",
    "BDBIAR_Source_Integrity_Status",
    "BDBIAR_OBJECTID_CANDIDATE",
    "BDBIAR_CLASS_CANDIDATE",
    "BDBIAR_ADDRESS_C_CANDIDATE",
    "Matched_POIs",
    "osmid",
    "building",
    "amenity",
    "shop",
    "OZP_ZONE_LABEL",
    "OZP_DESC_ENG",
    *STRUCTURE_SCOPE_FIELDS,
}

CLASSIFICATION_NUMERIC_FIELDS = {
    "ML_Confidence",
    "proba_住宅类别",
    "proba_商业类别",
    "proba_工业类别",
    "CATEGORY",
    "GROSSFLOORAREA",
    "BDBIAR_AGE",
    "BDBIAR_AGE_CANDIDATE",
    "distance_to_bdbiar",
}

RELEASE_GATE_FIELDS = {
    "Classification_Source",
    "LLM_Cache_Status",
    "Stage7_Repair_Action",
}

BDBIAR_ADOPTED_ATTRIBUTES = {
    "BDBIAR_CLASS",
    "BDBIAR_AGE",
    "BDBIAR_ADDRESS_C",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_blanks(frame: pd.DataFrame, geometry_name: str | None = None) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if column == geometry_name:
            continue
        if isinstance(result[column].dtype, pd.StringDtype) or result[column].dtype == object:
            blank = result[column].map(
                lambda value: isinstance(value, str) and value.strip() == ""
            )
            if blank.any():
                result[column] = result[column].mask(blank, pd.NA)
    return result


def _require_string_key(frame: pd.DataFrame, column: str, label: str) -> None:
    if column not in frame.columns:
        raise ValueError(f"{label} is missing key column {column}")
    values = frame[column]
    if not isinstance(values.dtype, pd.StringDtype):
        values = values.astype("string")
        frame[column] = values
    empty = values.isna() | values.str.strip().eq("")
    if empty.any():
        raise ValueError(f"{label} contains {int(empty.sum())} empty {column} values")
    padded = values.ne(values.str.strip())
    if padded.any():
        raise ValueError(f"{label} contains whitespace-padded {column} values")
    duplicates = values.duplicated(keep=False)
    if duplicates.any():
        examples = values.loc[duplicates].head(5).tolist()
        raise ValueError(f"{label} contains duplicate {column} values: {examples}")


def _read_classification(path: Path) -> pd.DataFrame:
    required = {
        RAW_ID,
        CSUID,
        *(source for _, source in PUBLIC_FIELDS if source not in PHYSICAL_CORE),
        *STRUCTURE_SCOPE_FIELDS,
        *RELEASE_GATE_FIELDS,
    }
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"Classification CSV is missing required columns: {missing}")
    frame = pd.read_csv(
        path,
        usecols=[column for column in header if column in required],
        dtype={column: "string" for column in CLASSIFICATION_STRING_FIELDS},
        keep_default_na=False,
        low_memory=False,
        encoding="utf-8-sig",
    )
    frame = _normalise_blanks(frame)
    for column in CLASSIFICATION_NUMERIC_FIELDS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    _require_string_key(frame, RAW_ID, "Classification CSV")
    _require_string_key(frame, CSUID, "Classification CSV")
    _validate_classification_provenance(frame)
    frame.attrs["source_column_count"] = len(header)
    return frame


def _validate_classification_provenance(frame: pd.DataFrame) -> None:
    match_quality = frame["BDBIAR_Match_Quality"].astype("string")
    usable = (
        frame["BDBIAR_Source_Attributes_Usable"]
        .astype("string")
        .str.casefold()
        .eq("true")
        .fillna(False)
    )
    non_unique = match_quality.ne("inside_unique").fillna(True)
    unsafe = non_unique | ~usable
    if frame.loc[unsafe, sorted(BDBIAR_ADOPTED_ATTRIBUTES)].notna().any().any():
        raise ValueError("Unsafe BDBIAR rows contain adopted source attributes")
    if frame.loc[unsafe, "BDBIAR_OBJECTID"].notna().any():
        raise ValueError("Unsafe BDBIAR matches contain an adopted record ID")

    source = frame["Classification_Source"].fillna("").astype(str)
    calibrated_source = frame["Calibrated_Source"].fillna("").astype(str)
    for label, values in (
        ("rule", source),
        ("calibrated", calibrated_source),
    ):
        bdbiar = values.str.contains("BDBIAR", regex=False)
        if (bdbiar & unsafe).any():
            raise ValueError(f"{label} classification cites unsafe BDBIAR evidence")
        if values.str.contains("GFA", case=False, regex=False).any():
            raise ValueError(f"{label} classification cites unusable GFA evidence")
        if values.str.contains("VLM", case=False, regex=False).any():
            raise ValueError(f"{label} classification cites an unbound VLM decision")

    llm = source.str.contains("LLM", case=False, regex=False)
    exact_status = frame["LLM_Cache_Status"].fillna("").isin(
        {"legacy_cache_exact_prompt", "native_cache_exact_prompt"}
    )
    if (llm & ~exact_status).any():
        raise ValueError("Published LLM classification lacks an exact current prompt contract")


def _read_physical(path: Path) -> gpd.GeoDataFrame:
    frame = gpd.read_parquet(path)
    if frame.crs is None or frame.crs.to_epsg() != 2326:
        raise ValueError(f"Verified physical GeoParquet CRS must be EPSG:2326, got {frame.crs}")
    if frame.geometry.name != GEOMETRY:
        raise ValueError(f"Verified physical geometry column must be named {GEOMETRY}")
    required = {ID, CSUID, GEOMETRY, *PHYSICAL_CORE}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Verified physical GeoParquet is missing required columns: {missing}")
    frame = _normalise_blanks(frame, GEOMETRY)
    _require_string_key(frame, ID, "Verified physical GeoParquet")
    _require_string_key(frame, CSUID, "Verified physical GeoParquet")
    invalid = frame.geometry.isna() | frame.geometry.is_empty | ~frame.geometry.is_valid
    if invalid.any():
        raise ValueError(f"Verified physical GeoParquet contains {int(invalid.sum())} invalid geometries")
    return frame


def _align(classification: pd.DataFrame, physical: gpd.GeoDataFrame) -> pd.DataFrame:
    class_ids = set(classification[RAW_ID])
    physical_ids = set(physical[ID])
    if class_ids != physical_ids:
        raise ValueError(
            "Classification and physical Building ID sets differ: "
            f"classification_only={len(class_ids - physical_ids)}, "
            f"physical_only={len(physical_ids - class_ids)}"
        )
    aligned = classification.set_index(RAW_ID).loc[physical[ID]].reset_index()
    aligned.index = physical.index
    mismatch = aligned[CSUID].ne(physical[CSUID])
    if mismatch.any():
        examples = physical.loc[mismatch, [ID, CSUID]].head(5).to_dict("records")
        raise ValueError(
            f"BUILDINGCSUID differs for {int(mismatch.sum())} Building IDs: {examples}"
        )
    return aligned


def assemble_frame(
    classification_path: Path, physical_path: Path
) -> tuple[
    gpd.GeoDataFrame,
    dict[str, str],
    dict[str, Any],
    pd.DataFrame,
    dict[str, Any],
]:
    classification = _read_classification(classification_path)
    classification, integrity_audit, integrity_summary = (
        classification_integrity.enforce_integrity(classification)
    )
    _, _, recheck = classification_integrity.enforce_integrity(classification)
    if recheck["changed_rows"]:
        raise AssertionError("Classification integrity output is not idempotent")
    physical = _read_physical(physical_path)
    input_stats = {
        "classification_csv": {
            "rows": int(len(classification)),
            "source_columns": int(classification.attrs["source_column_count"]),
            "consumed_columns": len(classification.columns),
            "consumed_column_names": classification.columns.tolist(),
            "consumed_null_counts": {
                column: int(classification[column].isna().sum())
                for column in classification.columns
            },
        },
        "verified_physical_geoparquet": {
            "rows": int(len(physical)),
            "columns_including_geometry": len(physical.columns),
            "column_names": physical.columns.tolist(),
            "null_counts": {
                column: int(physical[column].isna().sum()) for column in physical.columns
            },
            "crs": "EPSG:2326",
        },
    }
    aligned = _align(classification, physical)

    public: dict[str, pd.Series] = {}
    sources: dict[str, str] = {}
    for output, source in PUBLIC_FIELDS:
        if output == ID:
            public[output] = aligned[RAW_ID]
            sources[output] = (
                f"classification_csv:{RAW_ID}; exact-match validated against "
                f"verified_physical_attributes.parquet:{ID}"
            )
        elif output in PHYSICAL_CORE:
            public[output] = physical[output]
            sources[output] = f"verified_physical_attributes.parquet:{output}"
        else:
            public[output] = aligned[source]
            sources[output] = f"classification_csv:{source}"

    physical_extras = [
        column
        for column in physical.columns
        if column not in PHYSICAL_PUBLIC_FIELDS and column != GEOMETRY
    ]
    duplicate_outputs = set(public).intersection(physical_extras)
    if duplicate_outputs:
        raise ValueError(f"Physical columns collide with public fields: {sorted(duplicate_outputs)}")
    for column in physical_extras:
        public[column] = physical[column]
        sources[column] = f"verified_physical_attributes.parquet:{column}"
    for column in STRUCTURE_SCOPE_FIELDS:
        if column in public:
            raise ValueError(f"Structure scope field collides with another output: {column}")
        public[column] = aligned[column]
        sources[column] = f"classification_csv:{column}"

    attributes = pd.DataFrame(public, index=physical.index)
    result = gpd.GeoDataFrame(
        attributes,
        geometry=physical.geometry.copy(),
        crs=physical.crs,
    )
    if not result["UBEM_Main_Class_主类别"].equals(aligned["Calibrated_Main_Class"]):
        raise AssertionError("Assembler changed Calibrated_Main_Class")
    for column in PHYSICAL_CORE:
        if not result[column].equals(physical[column]):
            raise AssertionError(f"Assembler changed authoritative physical field {column}")
    return result, sources, input_stats, integrity_audit, integrity_summary


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            raise ValueError("Non-finite numeric value cannot be written to GeoJSON")
        return value
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _orient_rfc7946(geometry):
    if isinstance(geometry, Polygon):
        return orient(geometry, sign=1.0)
    if isinstance(geometry, MultiPolygon):
        return MultiPolygon([orient(part, sign=1.0) for part in geometry.geoms])
    if isinstance(geometry, GeometryCollection):
        return GeometryCollection([_orient_rfc7946(part) for part in geometry.geoms])
    return geometry


def _write_geojson(frame: gpd.GeoDataFrame, path: Path) -> None:
    geometries = frame.geometry.to_crs("EPSG:4326")
    attributes = frame.drop(columns=frame.geometry.name)
    names = attributes.columns.tolist()
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write('{"type":"FeatureCollection","features":[\n')
        first = True
        for values, geometry in zip(
            attributes.itertuples(index=False, name=None), geometries, strict=True
        ):
            if geometry is None or geometry.is_empty or not geometry.is_valid:
                raise ValueError("Geometry became invalid during WGS84 transformation")
            geometry = _orient_rfc7946(geometry)
            feature = {
                "type": "Feature",
                "geometry": mapping(geometry),
                "properties": {
                    name: _json_value(value) for name, value in zip(names, values, strict=True)
                },
            }
            if not first:
                stream.write(",\n")
            json.dump(
                feature,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            first = False
        stream.write("\n]}\n")


def _field_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean"
    if pd.api.types.is_integer_dtype(series.dtype):
        return "integer"
    if pd.api.types.is_float_dtype(series.dtype):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    present = series.dropna()
    if not present.empty and present.map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        return "boolean"
    return "string"


def _schema(frame: gpd.GeoDataFrame, sources: dict[str, str]) -> dict[str, Any]:
    attributes = [column for column in frame.columns if column != frame.geometry.name]
    return {
        "dataset": "HKBFETD",
        "version": "2.0",
        "row_semantics": "one building or structure component per row",
        "null_semantics": "unknown, unavailable, withdrawn, or not applicable; never zero",
        "geometry": {
            "column": frame.geometry.name,
            "geoparquet_crs": "EPSG:2326",
            "geojson_crs": "RFC 7946 WGS84 (EPSG:4326; no crs member)",
        },
        "fields": [
            {
                "name": column,
                "type": _field_type(frame[column]),
                "nullable": bool(frame[column].isna().any()),
                "null_count": int(frame[column].isna().sum()),
                "source": sources[column],
            }
            for column in attributes
        ],
    }


def _file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def assemble_release_dataset(
    classification_path: Path, physical_path: Path, output_dir: Path
) -> dict[str, Any]:
    classification_path = Path(classification_path).expanduser().resolve()
    physical_path = Path(physical_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    for label, path in (
        ("classification CSV", classification_path),
        ("verified physical GeoParquet", physical_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    classification_hash = sha256(classification_path)
    physical_hash = sha256(physical_path)
    frame, sources, input_stats, integrity_audit, integrity_summary = assemble_frame(
        classification_path, physical_path
    )
    attributes = [column for column in frame.columns if column != frame.geometry.name]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.assembling-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary) / output_dir.name
        staging.mkdir()
        csv_path = staging / "HKBFETD_v2.csv"
        parquet_path = staging / "HKBFETD_v2.parquet"
        geojson_path = staging / "HKBFETD_v2.geojson"
        schema_path = staging / "schema.json"
        integrity_audit_path = staging / "classification_integrity_audit.csv"
        integrity_manifest_path = staging / "classification_integrity_manifest.json"

        frame[attributes].to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig",
            na_rep="",
            quoting=csv.QUOTE_MINIMAL,
        )
        frame.to_parquet(parquet_path, index=False)
        _write_geojson(frame, geojson_path)
        schema_path.write_text(
            json.dumps(_schema(frame, sources), ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        integrity_audit.to_csv(
            integrity_audit_path, index=False, encoding="utf-8-sig", na_rep=""
        )
        integrity_manifest = {
            "status": "PASS",
            "policy": (
                "Preserve main classes; normalize exact frozen-taxonomy aliases; "
                "fail closed only on dependent subclass/mixture contradictions."
            ),
            **integrity_summary,
            "remaining_changed_rows_after_gate": 0,
            "classification_input_sha256": classification_hash,
            "taxonomy_sha256": sha256(classification_integrity.DEFAULT_TAXONOMY),
            "audit_sha256": sha256(integrity_audit_path),
        }
        integrity_manifest_path.write_text(
            json.dumps(integrity_manifest, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

        scope_mask = frame[list(STRUCTURE_SCOPE_FIELDS)].notna().any(axis=1)
        table_null_counts = {
            column: int(frame[column].isna().sum()) for column in attributes
        }
        common_table_stats = {
            "rows": int(len(frame)),
            "attribute_columns": len(attributes),
            "null_counts": table_null_counts,
        }
        manifest = {
            "dataset": "HKBFETD",
            "version": "2.0",
            "status": "assembled",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "classification_csv": {
                    "path": str(classification_path),
                    "sha256": classification_hash,
                    **input_stats["classification_csv"],
                },
                "verified_physical_geoparquet": {
                    "path": str(physical_path),
                    "sha256": physical_hash,
                    **input_stats["verified_physical_geoparquet"],
                },
            },
            "join_validation": {
                "key_columns": [ID, CSUID],
                "cardinality": "one_to_one",
                "rows": int(len(frame)),
                "unique_building_ids": int(frame[ID].nunique()),
                "unique_buildingcsuids": int(frame[CSUID].nunique()),
                "unmatched_rows": 0,
                "buildingcsuid_mismatches": 0,
                "classification_main_class_copied_without_rewrite": True,
                "authoritative_physical_core_fields": list(PHYSICAL_CORE),
                "structure_scope_rows_retained": int(scope_mask.sum()),
            },
            "classification_integrity": integrity_manifest,
            "table": {
                **common_table_stats,
                "columns_including_geometry": len(frame.columns),
            },
            "serialization": {
                "csv": "UTF-8-SIG; empty fields represent null",
                "geoparquet": "EPSG:2326; empty strings normalised to null",
                "geojson": "RFC 7946 WGS84; streamed; no crs member; no NaN/Infinity",
            },
            "outputs": {
                csv_path.name: {
                    **_file_record(csv_path),
                    **common_table_stats,
                    "geometry_columns": 0,
                },
                parquet_path.name: {
                    **_file_record(parquet_path),
                    **common_table_stats,
                    "geometry_columns": 1,
                    "columns_including_geometry": len(frame.columns),
                    "crs": "EPSG:2326",
                },
                geojson_path.name: {
                    **_file_record(geojson_path),
                    **common_table_stats,
                    "geometry_members": 1,
                    "crs": "RFC 7946 WGS84 (EPSG:4326; no crs member)",
                    "polygon_ring_orientation": "outer counterclockwise; inner clockwise",
                },
                schema_path.name: {
                    **_file_record(schema_path),
                    "field_records": len(attributes),
                },
                integrity_audit_path.name: {
                    **_file_record(integrity_audit_path),
                    "rows": int(len(integrity_audit)),
                },
                integrity_manifest_path.name: _file_record(integrity_manifest_path),
            },
            "manifest_self_hash": "omitted because a file cannot contain its own stable hash",
        }
        manifest_path = staging / "assembly_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for attempt in range(6):
            try:
                staging.replace(output_dir)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                # Windows indexing or antivirus may briefly retain a handle to
                # the newly written GeoJSON/Parquet files.
                time.sleep(2**attempt)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble HKBFETD V2.0 public tables from final classification and verified physical data"
    )
    parser.add_argument("--classification-csv", required=True, type=Path)
    parser.add_argument("--physical-parquet", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = assemble_release_dataset(
        args.classification_csv, args.physical_parquet, args.output_dir
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
