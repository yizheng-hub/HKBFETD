"""Compare V1.1 and corrected V2 classifications with exclusive attribution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


ID_COLUMNS = ("Building_ID_建筑ID", "BUILDINGSTRUCTUREID")
MAIN_COLUMNS = ("UBEM_Main_Class_主类别", "Calibrated_Main_Class", "Final_Main_Class")
SUB_COLUMNS = ("UBEM_Sub_Class_子类别", "Calibrated_Sub_Class", "Final_Sub_Class")
SOURCE_COLUMNS = (
    "Classification_Source_分类来源",
    "Calibrated_Source",
    "Classification_Source",
)
NAME_EN_COLUMNS = ("LandsD_Building_Name_EN_地政署英文名", "OFFICIALBUILDINGNAMEEN")
NAME_TC_COLUMNS = ("LandsD_Building_Name_TC_地政署中文名", "OFFICIALBUILDINGNAMETC")
UNKNOWN = "Unknown_未知类别"
UNKNOWN_TOKENS = {"", "unknown", "unknown_未知类别", "未知", "未知类别", "nan", "none", "null", "na", "n/a"}
ATTRIBUTION_PRIORITY = (
    "nonbuilding_scope",
    "gfa_withdrawal",
    "new_llm",
    "vlm",
    "ml_retraining_or_step10",
)
ATTRIBUTION_ZH = {
    "nonbuilding_scope": "非建筑结构范围",
    "gfa_withdrawal": "GFA 证据撤回",
    "new_llm": "原提示新 LLM 候选",
    "vlm": "冻结 VLM 继承/删除",
    "ml_retraining_or_step10": "机器学习重训或 Step 10",
}
OUTPUT_NAMES = {
    "json": "classification_comparison.json",
    "changes": "classification_changes.csv",
    "examples": "classification_typical_examples.csv",
    "markdown": "classification_comparison_zh.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _column(frame: pd.DataFrame, candidates: Iterable[str], role: str, *, required: bool = True) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    if required:
        raise ValueError(f"{role}缺少字段，候选字段为: {list(candidates)}")
    return None


def _normalise_id(series: pd.Series, role: str) -> pd.Series:
    result = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    if result.isna().any() or result.eq("").any():
        raise ValueError(f"{role}包含空建筑ID")
    if result.duplicated().any():
        examples = result[result.duplicated(keep=False)].head(5).tolist()
        raise ValueError(f"{role}包含重复建筑ID: {examples}")
    return result


def _label(value: object) -> str:
    if pd.isna(value):
        return UNKNOWN
    text = str(value).strip()
    return UNKNOWN if text.casefold() in UNKNOWN_TOKENS else text


def _source(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return "(空)"
    return str(value).strip()


def _read_version(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype="string", low_memory=False, encoding="utf-8-sig")
    id_column = _column(frame, ID_COLUMNS, f"{prefix}分类表")
    main_column = _column(frame, MAIN_COLUMNS, f"{prefix}分类表")
    sub_column = _column(frame, SUB_COLUMNS, f"{prefix}分类表")
    source_column = _column(frame, SOURCE_COLUMNS, f"{prefix}分类表")
    name_en = _column(frame, NAME_EN_COLUMNS, f"{prefix}分类表", required=False)
    name_tc = _column(frame, NAME_TC_COLUMNS, f"{prefix}分类表", required=False)
    result = pd.DataFrame(
        {
            "Building_ID_建筑ID": _normalise_id(frame[id_column], f"{prefix}分类表"),
            f"{prefix}_Main_Class": frame[main_column].map(_label),
            f"{prefix}_Sub_Class": frame[sub_column].map(_label),
            f"{prefix}_Classification_Source": frame[source_column].map(_source),
            f"{prefix}_Name_EN": frame[name_en].fillna("").str.strip() if name_en else "",
            f"{prefix}_Name_TC": frame[name_tc].fillna("").str.strip() if name_tc else "",
        }
    )
    return result


def _read_change_file(path: Path, role: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype="string", low_memory=False, encoding="utf-8-sig")
    id_column = _column(frame, ID_COLUMNS, role)
    frame = frame.copy()
    frame["_id"] = _normalise_id(frame[id_column], role)
    return frame


def _action(frame: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    column = _column(frame, candidates, "变化表", required=False)
    if column is None:
        return pd.Series("", index=frame.index, dtype="string")
    return frame[column].fillna("").astype("string").str.strip()


def _id_sort(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_numeric_id"] = pd.to_numeric(result["Building_ID_建筑ID"], errors="coerce")
    result["_text_id"] = result["Building_ID_建筑ID"].astype(str)
    result = result.sort_values(
        ["_numeric_id", "_text_id"], na_position="last", kind="stable"
    )
    return result.drop(columns=["_numeric_id", "_text_id"])


def _transition_type(row: pd.Series) -> str:
    if not row["Main_Changed"]:
        return "main_unchanged_sub_changed"
    old_unknown = row["Old_Main_Class"] == UNKNOWN
    new_unknown = row["New_Main_Class"] == UNKNOWN
    if old_unknown and not new_unknown:
        return "unknown_to_known"
    if not old_unknown and new_unknown:
        return "known_to_unknown"
    return "known_to_different_known"


def _transitions(frame: pd.DataFrame, old: str, new: str) -> list[dict]:
    grouped = (
        frame.groupby([old, new], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={old: "from", new: "to"})
    )
    grouped["changed"] = grouped["from"] != grouped["to"]
    grouped = grouped.sort_values(["count", "from", "to"], ascending=[False, True, True])
    return grouped.to_dict(orient="records")


def _old_source_groups(frame: pd.DataFrame) -> list[dict]:
    records = []
    for source, group in frame.groupby("Old_Classification_Source", dropna=False):
        old_unknown = group["Old_Main_Class"].eq(UNKNOWN)
        new_unknown = group["New_Main_Class"].eq(UNKNOWN)
        records.append(
            {
                "old_classification_source": source,
                "rows": int(len(group)),
                "main_changed": int(group["Main_Changed"].sum()),
                "sub_changed": int(group["Sub_Changed"].sum()),
                "either_changed": int(group["Either_Changed"].sum()),
                "main_unknown_to_known": int((old_unknown & ~new_unknown).sum()),
                "main_known_to_unknown": int((~old_unknown & new_unknown).sum()),
                "contains_keyword": "keyword" in str(source).casefold(),
            }
        )
    return sorted(records, key=lambda row: (-row["rows"], str(row["old_classification_source"])))


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("|" + "|".join(clean(value) for value in row) + "|" for row in rows)
    return "\n".join(lines)


def _write_markdown(summary: dict, path: Path) -> None:
    counts = summary["counts"]
    unknown = summary["unknown_known"]
    gate = summary["gates"]["old_keyword_main_class_stability"]
    attribution_rows = [
        [ATTRIBUTION_ZH[key], summary["attribution_counts"].get(key, 0)]
        for key in ATTRIBUTION_PRIORITY
    ]
    main_changed = [row for row in summary["main_transitions"] if row["changed"]]
    source_rows = summary["old_source_groups"][:20]
    lines = [
        "# V1.1 → 修正 V2 分类对比",
        "",
        f"**状态：{summary['status']}**",
        "",
        f"共同建筑记录：{counts['common_rows']:,}；主类变化：{counts['main_changed']:,}；"
        f"子类变化：{counts['sub_changed']:,}；任一分类字段变化：{counts['either_changed']:,}。",
        "",
        f"旧来源含 Keyword 的记录为 {gate['rows']:,} 条，主类变化 {gate['violations']:,} 条。"
        + ("硬门通过。" if gate["status"] == "PASS" else "硬门失败，当前结果不得发布。"),
        "",
        "## Unknown 与已知类别转换",
        "",
        _markdown_table(
            ["范围", "Unknown→已知", "已知→Unknown"],
            [
                ["主类", unknown["main_unknown_to_known"], unknown["main_known_to_unknown"]],
                ["子类", unknown["sub_unknown_to_known"], unknown["sub_known_to_unknown"]],
            ],
        ),
        "",
        "## 互斥归因",
        "",
        "优先级固定为：非建筑结构范围 → GFA 证据撤回 → 原提示新 LLM 候选 → 冻结 VLM → 机器学习重训或 Step 10。每条变化只计一次。",
        "",
        _markdown_table(["归因", "变化记录"], attribution_rows),
        "",
        "## 主类变化转移",
        "",
        _markdown_table(
            ["V1.1", "修正 V2", "数量"],
            [[row["from"], row["to"], row["count"]] for row in main_changed],
        ),
        "",
        "## 旧 Classification_Source 分组（前 20 组）",
        "",
        _markdown_table(
            ["旧来源", "记录", "主类变化", "子类变化", "任一变化"],
            [
                [
                    row["old_classification_source"],
                    row["rows"],
                    row["main_changed"],
                    row["sub_changed"],
                    row["either_changed"],
                ]
                for row in source_rows
            ],
        ),
        "",
        f"全部逐栋变化见 `{OUTPUT_NAMES['changes']}`；确定性抽取的典型例子见 `{OUTPUT_NAMES['examples']}`；完整转移和来源分组见 `{OUTPUT_NAMES['json']}`。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def compare_and_write(
    v1_path: Path,
    v2_path: Path,
    stage3_gfa_path: Path,
    stage5_path: Path,
    stage7_path: Path,
    out_dir: Path,
    *,
    examples_per_group: int = 3,
) -> dict:
    paths = [Path(path).resolve() for path in (v1_path, v2_path, stage3_gfa_path, stage5_path, stage7_path)]
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"缺少输入文件: {missing}")
    if examples_per_group < 1:
        raise ValueError("examples_per_group必须至少为1")

    old = _read_version(paths[0], "Old")
    new = _read_version(paths[1], "New")
    old_ids = set(old["Building_ID_建筑ID"])
    new_ids = set(new["Building_ID_建筑ID"])
    if old_ids != new_ids:
        raise ValueError(
            "V1.1与V2建筑ID集合不同: "
            f"仅V1.1={len(old_ids - new_ids)}, 仅V2={len(new_ids - old_ids)}"
        )
    frame = old.merge(new, on="Building_ID_建筑ID", validate="one_to_one")
    frame["Name_EN"] = frame["New_Name_EN"].where(frame["New_Name_EN"].ne(""), frame["Old_Name_EN"])
    frame["Name_TC"] = frame["New_Name_TC"].where(frame["New_Name_TC"].ne(""), frame["Old_Name_TC"])
    frame["Main_Changed"] = frame["Old_Main_Class"] != frame["New_Main_Class"]
    frame["Sub_Changed"] = frame["Old_Sub_Class"] != frame["New_Sub_Class"]
    frame["Either_Changed"] = frame["Main_Changed"] | frame["Sub_Changed"]

    stage3 = _read_change_file(paths[2], "Stage3 GFA变化表")
    stage5 = _read_change_file(paths[3], "Stage5变化表")
    stage7 = _read_change_file(paths[4], "Stage7变化表")
    gfa_ids = set(stage3["_id"])
    stage5_action = _action(stage5, ("Stage5_Repair_Action", "Repair_Action", "Action"))
    new_llm_mask = stage5_action.str.contains(
        "native_llm_override_new_candidate_without_legacy_cache", case=False, regex=False
    )
    new_llm_ids = set(stage5.loc[new_llm_mask, "_id"])
    stage7_action = _action(stage7, ("Stage7_Repair_Action", "Repair_Action", "Action"))
    nonbuilding_mask = stage7_action.str.contains(
        "verified_nonbuilding_structure_excluded_before_ML", case=False, regex=False
    )
    vlm_mask = stage7_action.str.startswith("inherited_from_") | stage7_action.eq(
        "deleted_by_frozen_vlm_decision"
    )
    unrecognised = stage7_action.ne("") & ~(nonbuilding_mask | vlm_mask)
    if unrecognised.any():
        raise ValueError(
            "Stage7变化表含无法归因的动作: "
            f"{sorted(stage7_action[unrecognised].unique().tolist())[:10]}"
        )
    nonbuilding_ids = set(stage7.loc[nonbuilding_mask, "_id"])
    vlm_ids = set(stage7.loc[vlm_mask, "_id"])

    frame["In_Stage3_GFA"] = frame["Building_ID_建筑ID"].isin(gfa_ids)
    frame["In_Stage5_New_LLM"] = frame["Building_ID_建筑ID"].isin(new_llm_ids)
    frame["In_Stage7_VLM"] = frame["Building_ID_建筑ID"].isin(vlm_ids)
    frame["In_Stage7_Nonbuilding"] = frame["Building_ID_建筑ID"].isin(nonbuilding_ids)
    frame["Attribution"] = "ml_retraining_or_step10"
    frame.loc[frame["In_Stage7_VLM"], "Attribution"] = "vlm"
    frame.loc[frame["In_Stage5_New_LLM"], "Attribution"] = "new_llm"
    frame.loc[frame["In_Stage3_GFA"], "Attribution"] = "gfa_withdrawal"
    frame.loc[frame["In_Stage7_Nonbuilding"], "Attribution"] = "nonbuilding_scope"

    changed = frame.loc[frame["Either_Changed"]].copy()
    changed["Main_Transition_Type"] = changed.apply(_transition_type, axis=1)
    changed["Attribution_zh"] = changed["Attribution"].map(ATTRIBUTION_ZH)
    output_columns = [
        "Building_ID_建筑ID",
        "Name_EN",
        "Name_TC",
        "Old_Main_Class",
        "New_Main_Class",
        "Main_Changed",
        "Old_Sub_Class",
        "New_Sub_Class",
        "Sub_Changed",
        "Old_Classification_Source",
        "New_Classification_Source",
        "Main_Transition_Type",
        "Attribution",
        "Attribution_zh",
        "In_Stage7_Nonbuilding",
        "In_Stage3_GFA",
        "In_Stage5_New_LLM",
        "In_Stage7_VLM",
    ]
    changed = _id_sort(changed[output_columns])

    examples = []
    transition_order = (
        "unknown_to_known",
        "known_to_unknown",
        "known_to_different_known",
        "main_unchanged_sub_changed",
    )
    for attribution in ATTRIBUTION_PRIORITY:
        for transition in transition_order:
            group = changed.loc[
                changed["Attribution"].eq(attribution)
                & changed["Main_Transition_Type"].eq(transition)
            ]
            if not group.empty:
                examples.append(_id_sort(group).head(examples_per_group))
    typical = pd.concat(examples, ignore_index=True) if examples else changed.head(0).copy()

    keyword_mask = frame["Old_Classification_Source"].str.contains(
        "keyword", case=False, regex=False, na=False
    )
    keyword_violations = frame.loc[keyword_mask & frame["Main_Changed"]]
    status = "PASS" if keyword_violations.empty else "FAIL"
    old_main_unknown = frame["Old_Main_Class"].eq(UNKNOWN)
    new_main_unknown = frame["New_Main_Class"].eq(UNKNOWN)
    old_sub_unknown = frame["Old_Sub_Class"].eq(UNKNOWN)
    new_sub_unknown = frame["New_Sub_Class"].eq(UNKNOWN)
    attribution_counts = changed["Attribution"].value_counts().reindex(ATTRIBUTION_PRIORITY, fill_value=0)
    if int(attribution_counts.sum()) != len(changed):
        raise AssertionError("互斥归因计数未覆盖全部变化记录")

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    changes_path = out_dir / OUTPUT_NAMES["changes"]
    examples_path = out_dir / OUTPUT_NAMES["examples"]
    markdown_path = out_dir / OUTPUT_NAMES["markdown"]
    json_path = out_dir / OUTPUT_NAMES["json"]
    changed.to_csv(changes_path, index=False, encoding="utf-8-sig")
    typical.to_csv(examples_path, index=False, encoding="utf-8-sig")

    summary = {
        "schema_version": 1,
        "status": status,
        "counts": {
            "v1_rows": int(len(old)),
            "v2_rows": int(len(new)),
            "common_rows": int(len(frame)),
            "main_changed": int(frame["Main_Changed"].sum()),
            "sub_changed": int(frame["Sub_Changed"].sum()),
            "either_changed": int(frame["Either_Changed"].sum()),
            "unchanged_main_and_sub": int((~frame["Either_Changed"]).sum()),
        },
        "unknown_known": {
            "main_unknown_to_known": int((old_main_unknown & ~new_main_unknown).sum()),
            "main_known_to_unknown": int((~old_main_unknown & new_main_unknown).sum()),
            "sub_unknown_to_known": int((old_sub_unknown & ~new_sub_unknown).sum()),
            "sub_known_to_unknown": int((~old_sub_unknown & new_sub_unknown).sum()),
        },
        "gates": {
            "old_keyword_main_class_stability": {
                "status": status,
                "rows": int(keyword_mask.sum()),
                "violations": int(len(keyword_violations)),
                "violation_ids": keyword_violations["Building_ID_建筑ID"].tolist(),
                "rule": "Every V1.1 row whose Classification_Source contains Keyword must keep its main class.",
            }
        },
        "attribution_priority": list(ATTRIBUTION_PRIORITY),
        "attribution_counts": {key: int(value) for key, value in attribution_counts.items()},
        "main_transitions": _transitions(frame, "Old_Main_Class", "New_Main_Class"),
        "sub_transitions": _transitions(frame, "Old_Sub_Class", "New_Sub_Class"),
        "old_source_groups": _old_source_groups(frame),
        "inputs": {
            role: {"path": str(path), "sha256": sha256(path)}
            for role, path in zip(("v1_1", "v2_corrected", "stage3_gfa", "stage5_changes", "stage7_changes"), paths)
        },
        "outputs": {
            "changes_csv": str(changes_path),
            "typical_examples_csv": str(examples_path),
            "markdown": str(markdown_path),
        },
    }
    _write_markdown(summary, markdown_path)
    summary["outputs"]["sha256"] = {
        changes_path.name: sha256(changes_path),
        examples_path.name: sha256(examples_path),
        markdown_path.name: sha256(markdown_path),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", required=True, type=Path, help="V1.1 public CSV")
    parser.add_argument("--v2", required=True, type=Path, help="corrected V2 public CSV")
    parser.add_argument("--stage3-gfa", required=True, type=Path)
    parser.add_argument("--stage5", required=True, type=Path)
    parser.add_argument("--stage7", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--examples-per-group", type=int, default=3)
    args = parser.parse_args()
    summary = compare_and_write(
        args.v1,
        args.v2,
        args.stage3_gfa,
        args.stage5,
        args.stage7,
        args.out_dir,
        examples_per_group=args.examples_per_group,
    )
    print(json.dumps({"status": summary["status"], **summary["counts"]}, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
