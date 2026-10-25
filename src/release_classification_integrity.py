"""Validate release classification dependencies without reclassifying buildings.

The frozen original pipeline decides the main class before this gate runs.
This module therefore treats ``Calibrated_Main_Class`` as immutable.  It only
normalizes exact aliases from the frozen taxonomy and withholds dependent
subclass or mixture values that cannot be reconciled with that main class.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


ID_COLUMN = "BUILDINGSTRUCTUREID"
MAIN_COLUMN = "Calibrated_Main_Class"
SUB_COLUMN = "Calibrated_Sub_Class"
MIX_COLUMN = "Calibrated_Mix_Proportion"
UNKNOWN_SUBCLASS = "Unknown_未知类别"
DEFAULT_TAXONOMY = (
    Path(__file__).resolve().parent
    / "frozen_original_method"
    / "ctl"
    / "llm_taxonomy.json"
)

MAIN_BY_TAXONOMY_KEY = {
    "住宅类别": "Residential_住宅类别",
    "商业类别": "Commercial_商业类别",
    "工业类别": "Industrial_工业类别",
    "非评估类别": "Non-assessed_非评估类别",
}
ALLOWED_MAIN_CLASSES = {
    *MAIN_BY_TAXONOMY_KEY.values(),
    "Mixed-use_混合用途",
}
BROAD_PARENT_LABELS = {
    "Residential_住宅类别",
    "Commercial_商业类别",
    "Industrial_工业类别",
}
NON_ASSESSED_LABELS = {
    "绝对噪声相关": "Noise-related Infrastructure_噪声相关基础设施",
    "交通基础设施": "Transport Infrastructure_交通基础设施",
    "临时/杂项设施": "Temporary/Miscellaneous Structures_临时/杂项设施",
}
RELEASE_LABEL_PATCHES = {
    "Absolute Noise_绝对噪声相关": "Noise-related Infrastructure_噪声相关基础设施",
    "Temporary/Misc Facilities_临时/杂项设施": (
        "Temporary/Miscellaneous Structures_临时/杂项设施"
    ),
}
PROVEN_RENDERER_ALIASES = {
    "Non-Non-manufacturing_非制造业": "Non-manufacturing_非制造业",
}
UNKNOWN_ALIASES = {
    "",
    "Unknown",
    "unknown",
    "未知类别",
    "Unknown_未知类别",
    "Unknown (未知类别)",
    "Unknown Mixed (未知混合)",
    "未知混合",
}
NUMBER = re.compile(r"(?:0|[1-9]\d*)(?:\.\d+)?")
TAXONOMY_LABEL = re.compile(
    r"\s*([^()（）]+?)\s*[（(]\s*([^()（）]+?)\s*[)）]\s*"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def add_alias(
    candidates: dict[str, set[str]], alias: str, canonical: str
) -> None:
    alias = alias.strip()
    if alias:
        candidates.setdefault(alias, set()).add(canonical)


def build_taxonomy(
    taxonomy_path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build exact alias and canonical-parent maps from the frozen taxonomy."""
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8-sig"))
    aliases: dict[str, set[str]] = {}
    parent_by_label: dict[str, str] = {}

    for taxonomy_main, raw_labels in taxonomy.items():
        if taxonomy_main == "__STRONG_KEYWORDS__":
            continue
        if taxonomy_main not in MAIN_BY_TAXONOMY_KEY:
            raise ValueError(f"Unsupported taxonomy main class: {taxonomy_main}")
        parent = MAIN_BY_TAXONOMY_KEY[taxonomy_main]
        if not isinstance(raw_labels, list):
            raise ValueError(f"Taxonomy labels must be a list: {taxonomy_main}")
        for raw_value in raw_labels:
            raw_label = str(raw_value).strip()
            match = TAXONOMY_LABEL.fullmatch(raw_label)
            if match is not None:
                chinese, english = match.group(1).strip(), match.group(2).strip()
                internal = f"{english}_{chinese}"
                canonical = RELEASE_LABEL_PATCHES.get(internal, internal)
                label_aliases = {raw_label, chinese, english, internal, canonical}
            else:
                if taxonomy_main != "非评估类别" or raw_label not in NON_ASSESSED_LABELS:
                    raise ValueError(f"Unrecognized taxonomy label: {raw_label}")
                canonical = NON_ASSESSED_LABELS[raw_label]
                label_aliases = {raw_label, canonical}
                label_aliases.update(
                    alias
                    for alias, patched in RELEASE_LABEL_PATCHES.items()
                    if patched == canonical
                )
            parent_by_label[canonical] = parent
            for alias in label_aliases:
                add_alias(aliases, alias, canonical)

    for alias, canonical in PROVEN_RENDERER_ALIASES.items():
        if canonical not in parent_by_label:
            raise ValueError(f"Renderer alias target is absent from taxonomy: {canonical}")
        add_alias(aliases, alias, canonical)

    for parent in BROAD_PARENT_LABELS:
        add_alias(aliases, parent, parent)
        parent_by_label[parent] = parent

    ambiguous = {
        alias: sorted(canonical)
        for alias, canonical in aliases.items()
        if len(canonical) != 1
    }
    if ambiguous:
        raise ValueError(f"Ambiguous taxonomy aliases: {ambiguous}")
    return (
        {alias: next(iter(canonical)) for alias, canonical in aliases.items()},
        parent_by_label,
    )


def segment_labels(
    value: str, aliases: dict[str, str]
) -> list[tuple[str, str]] | None:
    """Parse hyphen-delimited labels without splitting lexical hyphens."""
    labels = sorted(aliases, key=len, reverse=True)

    @lru_cache(maxsize=None)
    def visit(offset: int) -> tuple[tuple[str, str], ...] | None:
        for alias in labels:
            if not value.startswith(alias, offset):
                continue
            end = offset + len(alias)
            item = (alias, aliases[alias])
            if end == len(value):
                return (item,)
            if value[end : end + 1] == "-":
                tail = visit(end + 1)
                if tail is not None:
                    return (item, *tail)
        return None

    result = visit(0)
    return list(result) if result is not None else None


def parse_mixture(
    value: str, aliases: dict[str, str]
) -> tuple[list[tuple[str, str, float]] | None, str | None]:
    """Parse an exact ``label:weight-label:weight`` mixture."""
    labels = sorted(aliases, key=len, reverse=True)

    @lru_cache(maxsize=None)
    def visit(offset: int) -> tuple[tuple[str, str, float], ...] | None:
        for alias in labels:
            prefix = f"{alias}:"
            if not value.startswith(prefix, offset):
                continue
            number_match = NUMBER.match(value, offset + len(prefix))
            if number_match is None:
                continue
            number_text = number_match.group(0)
            item = (aliases[alias], number_text, float(number_text))
            end = number_match.end()
            if end == len(value):
                return (item,)
            if value[end : end + 1] == "-":
                tail = visit(end + 1)
                if tail is not None:
                    return (item, *tail)
        return None

    parsed = visit(0)
    if parsed is None:
        return None, "non_whitelist_or_invalid_syntax"
    rows = list(parsed)
    labels_in_ratio = [label for label, _, _ in rows]
    weights = [weight for _, _, weight in rows]
    if any(not math.isfinite(weight) or weight <= 0 or weight > 1 for weight in weights):
        return None, "invalid_weight"
    if len(labels_in_ratio) != len(set(labels_in_ratio)):
        return None, "duplicate_label"
    if abs(sum(weights) - 1.0) > 0.031:
        return None, "sum_not_one"
    return rows, None


def enforce_integrity(
    source: pd.DataFrame, taxonomy_path: Path = DEFAULT_TAXONOMY
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return a sanitized copy, a changed-row ledger, and summary counts."""
    required = {ID_COLUMN, MAIN_COLUMN, SUB_COLUMN, MIX_COLUMN}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    result = source.copy(deep=True)
    ids = result[ID_COLUMN].map(text)
    if ids.eq("").any() or not ids.is_unique:
        raise ValueError("Building IDs must be nonempty and unique")

    original_main = result[MAIN_COLUMN].copy(deep=True)
    main_labels = result[MAIN_COLUMN].map(text)
    unsupported_main = sorted(set(main_labels) - ALLOWED_MAIN_CLASSES)
    if unsupported_main:
        raise ValueError(f"Unsupported main-class labels: {unsupported_main}")

    aliases, parent_by_label = build_taxonomy(taxonomy_path)
    input_sub = result[SUB_COLUMN].copy(deep=True)
    input_mix = result[MIX_COLUMN].copy(deep=True)
    output_sub: list[Any] = []
    output_mix: list[Any] = []
    row_actions: list[str] = []
    action_counts: Counter[str] = Counter()

    for main_label, sub_value, mix_value in zip(
        main_labels, input_sub, input_mix
    ):
        sub_text = text(sub_value)
        mix_text = text(mix_value)
        actions: list[str] = []
        sub_failed = False

        if sub_text in UNKNOWN_ALIASES:
            sub_out = UNKNOWN_SUBCLASS
            if sub_text != UNKNOWN_SUBCLASS:
                actions.append("subclass_unknown_normalized")
            sub_failed = True
        else:
            sub_parts = segment_labels(sub_text, aliases)
            if sub_parts is None:
                sub_out = UNKNOWN_SUBCLASS
                actions.append("subclass_withheld_non_whitelist")
                sub_failed = True
            else:
                canonical_subs = [canonical for _, canonical in sub_parts]
                sub_out = "-".join(canonical_subs)
                if len(canonical_subs) != len(set(canonical_subs)):
                    sub_out = UNKNOWN_SUBCLASS
                    actions.append("subclass_withheld_duplicate_label")
                    sub_failed = True
                elif main_label != "Mixed-use_混合用途":
                    parents = {parent_by_label[label] for label in canonical_subs}
                    if len(canonical_subs) != 1 or parents != {main_label}:
                        sub_out = UNKNOWN_SUBCLASS
                        actions.append("subclass_withheld_parent_conflict")
                        sub_failed = True
                if not sub_failed and sub_out != sub_text:
                    actions.append("subclass_alias_normalized")

        ratio_reason: str | None = None
        empty_mixture = mix_text.casefold() in {
            "",
            "nan",
            "none",
            "null",
            "unknown",
            UNKNOWN_SUBCLASS.casefold(),
        }
        if empty_mixture:
            mix_out: Any = pd.NA
            if mix_text:
                actions.append("mixture_unknown_normalized")
        else:
            parsed_ratio, ratio_reason = parse_mixture(mix_text, aliases)
            if ratio_reason is None and parsed_ratio is not None:
                ratio_parents = {
                    parent_by_label[label] for label, _, _ in parsed_ratio
                }
                if main_label == "Mixed-use_混合用途":
                    if len(ratio_parents) < 2:
                        ratio_reason = "mixed_single_parent"
                elif ratio_parents != {main_label}:
                    ratio_reason = "parent_conflict"
            if ratio_reason is not None:
                mix_out = pd.NA
                actions.append(f"mixture_withheld_{ratio_reason}")
            else:
                assert parsed_ratio is not None
                mix_out = "-".join(
                    f"{label}:{number_text}"
                    for label, number_text, _ in parsed_ratio
                )
                if mix_out != mix_text:
                    actions.append("mixture_alias_normalized")

        if sub_failed and mix_text and not pd.isna(mix_out):
            mix_out = pd.NA
            actions = [
                action
                for action in actions
                if action != "mixture_alias_normalized"
            ]
            actions.append("mixture_withheld_due_subclass")

        if ratio_reason == "mixed_single_parent" and sub_out != UNKNOWN_SUBCLASS:
            sub_out = UNKNOWN_SUBCLASS
            actions = [
                action
                for action in actions
                if action != "subclass_alias_normalized"
            ]
            actions.append("subclass_withheld_mixed_single_parent")

        unique_actions = list(dict.fromkeys(actions))
        action_counts.update(unique_actions)
        output_sub.append(sub_out)
        output_mix.append(mix_out)
        row_actions.append(";".join(unique_actions))

    result[SUB_COLUMN] = output_sub
    result[MIX_COLUMN] = output_mix
    if not result[MAIN_COLUMN].equals(original_main):
        raise AssertionError("Classification integrity gate changed a main class")
    for column in source.columns:
        if column in {SUB_COLUMN, MIX_COLUMN}:
            continue
        if not result[column].equals(source[column]):
            raise AssertionError(
                f"Classification integrity gate changed protected column: {column}"
            )

    changed = pd.Series(
        [bool(actions) for actions in row_actions], index=result.index
    )
    audit = pd.DataFrame(
        {
            ID_COLUMN: source.loc[changed, ID_COLUMN].to_numpy(),
            MAIN_COLUMN: result.loc[changed, MAIN_COLUMN].to_numpy(),
            f"Input_{SUB_COLUMN}": input_sub.loc[changed].to_numpy(),
            f"Output_{SUB_COLUMN}": result.loc[changed, SUB_COLUMN].to_numpy(),
            f"Input_{MIX_COLUMN}": input_mix.loc[changed].to_numpy(),
            f"Output_{MIX_COLUMN}": result.loc[changed, MIX_COLUMN].to_numpy(),
            "Release_Integrity_Actions": [
                actions for actions in row_actions if actions
            ],
        }
    )
    summary = {
        "rows": int(len(result)),
        "main_classes_changed": 0,
        "changed_rows": int(changed.sum()),
        "alias_normalization_rows": int(
            sum("alias_normalized" in actions for actions in row_actions)
        ),
        "fail_closed_rows": int(
            sum("withheld" in actions for actions in row_actions)
        ),
        "action_counts": dict(sorted(action_counts.items())),
        "canonical_subclass_labels": sorted(parent_by_label),
    }
    return result, audit, summary


def run_gate(
    input_path: Path,
    output_path: Path,
    audit_path: Path,
    manifest_path: Path,
    taxonomy_path: Path = DEFAULT_TAXONOMY,
) -> dict[str, Any]:
    paths = [
        input_path.resolve(),
        output_path.resolve(),
        audit_path.resolve(),
        manifest_path.resolve(),
        taxonomy_path.resolve(),
    ]
    if len(set(paths)) != len(paths):
        raise ValueError(
            "Input, output, audit, manifest, and taxonomy paths must be distinct"
        )
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not taxonomy_path.is_file():
        raise FileNotFoundError(taxonomy_path)

    frame = pd.read_csv(
        input_path,
        dtype={ID_COLUMN: "string"},
        encoding="utf-8-sig",
        low_memory=False,
    )
    result, audit, summary = enforce_integrity(frame, taxonomy_path)
    for path in (output_path, audit_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")

    manifest = {
        "status": "PASS",
        "policy": (
            "Preserve main classes; normalize exact frozen-taxonomy aliases; "
            "fail closed only on dependent subclass/mixture contradictions."
        ),
        **summary,
        "input": str(input_path.resolve()),
        "input_sha256": sha256(input_path),
        "taxonomy": str(taxonomy_path.resolve()),
        "taxonomy_sha256": sha256(taxonomy_path),
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "audit": str(audit_path.resolve()),
        "audit_sha256": sha256(audit_path),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize exact classification aliases and fail closed on "
            "dependent taxonomy contradictions without changing main classes."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    args = parser.parse_args()
    manifest = run_gate(
        args.input,
        args.output,
        args.audit,
        args.manifest,
        args.taxonomy,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
