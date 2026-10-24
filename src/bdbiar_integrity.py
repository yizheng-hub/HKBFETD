"""Source-integrity controls for BDBIAR attribute donation."""

import pandas as pd


BLOCK_ID = "BDBIAR_Source_Block_ID"
STATUS = "BDBIAR_Source_Integrity_Status"
QA = "BDBIAR_Source_Integrity_QA"
EVIDENCE = "BDBIAR_Source_Integrity_Evidence"


def load_bdbiar_integrity_registry(path):
    registry = pd.read_csv(path, dtype={"stable_block_id": "string"}).fillna("")
    if registry["stable_block_id"].duplicated().any():
        raise ValueError("Duplicate stable Block ID in BDBIAR integrity registry")
    return registry.set_index("stable_block_id").to_dict("index")


def apply_bdbiar_integrity_guard(frame, registry):
    """Quarantine only registered Block IDs; preserve source candidates and geometry."""
    frame = frame.copy()
    if BLOCK_ID not in frame:
        frame[BLOCK_ID] = None
    if STATUS not in frame:
        frame[STATUS] = ""
    if QA not in frame:
        frame[QA] = ""
    if EVIDENCE not in frame:
        frame[EVIDENCE] = ""
    if "BDBIAR_Source_Attributes_Usable" not in frame:
        frame["BDBIAR_Source_Attributes_Usable"] = True

    if "BDBIAR_Match_Quality" in frame:
        non_unique = frame["BDBIAR_Match_Quality"].ne("inside_unique")
        _quarantine_attributes(frame, non_unique)
        frame.loc[non_unique, "BDBIAR_Source_Attributes_Usable"] = False
        missing_status = non_unique & frame[STATUS].eq("")
        frame.loc[missing_status, STATUS] = "match_quality_not_inside_unique"
        frame.loc[missing_status, QA] = "BD attributes are not donated unless the point match is inside_unique."

    stable_ids = frame[BLOCK_ID].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    for stable_id, row in registry.items():
        mask = stable_ids.eq(str(stable_id)).fillna(False)
        if not mask.any():
            continue
        status = row["status"]
        frame.loc[mask, STATUS] = status
        frame.loc[mask, QA] = row["rationale"]
        frame.loc[mask, EVIDENCE] = row["evidence_locator"]
        suppress = row["suppress_bd_donations"]
        if isinstance(suppress, str):
            suppress = suppress.strip().lower() == "true"
        if suppress:
            _quarantine_attributes(frame, mask)
            frame.loc[mask, "BDBIAR_Source_Attributes_Usable"] = False
    return frame


def _quarantine_attributes(frame, mask):
    for source_column, candidate_column in (
        ("BDBIAR_AGE", "BDBIAR_AGE_CANDIDATE"),
        ("BDBIAR_CLASS", "BDBIAR_CLASS_CANDIDATE"),
    ):
        if source_column not in frame:
            continue
        if candidate_column not in frame:
            frame[candidate_column] = float('nan') if source_column == 'BDBIAR_AGE' else None
        if source_column == 'BDBIAR_CLASS':
            frame[source_column] = frame[source_column].astype(object)
        for index in frame.index[mask]:
            new_source = frame.at[index, source_column]
            if pd.notna(new_source):
                frame.at[index, candidate_column] = new_source
        frame.loc[mask, source_column] = None
