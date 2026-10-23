"""Replay source-scoped HA block evidence offline from normalized snapshots.

The snapshots, district coverage and checksum manifest accompany the release's
attribute_source_evidence.zip. Names produce candidates, not verified components.
No heights, total storeys, GFA or primary classification fields are imputed.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import unicodedata
import pandas as pd
import pyarrow.parquet as pq

DISTRICTS={
    "a": "CENTRAL & WESTERN",
    "b": "WAN CHAI",
    "c": "EASTERN",
    "d": "SOUTHERN",
    "e": "YAU TSIM MONG",
    "f": "SHAM SHUI PO",
    "g": "KOWLOON CITY",
    "h": "WONG TAI SIN",
    "j": "KWUN TONG",
    "k": "TSUEN WAN",
    "l": "TUEN MUN",
    "m": "YUEN LONG",
    "n": "NORTH",
    "p": "TAI PO",
    "q": "SAI KUNG",
    "r": "SHA TIN",
    "s": "KWAI TSING",
    "t": "ISLANDS"
}

def gz_path(code): return HERE/f"{code}_normalized_snapshot.json.gz"

def load_cache(code):
    path=gz_path(code)
    with gzip.open(path,"rb") as f: raw=f.read()
    return raw, json.loads(raw)

def norm(value):
    value=unicodedata.normalize("NFKC",str(value or "")).casefold()
    return "".join(ch for ch in value if ch.isalnum())

def aggregate():
    blocks=[]; all_rows=[]; coverage=[]
    for letter,district in DISTRICTS.items():
        code="ha_prhs_"+letter; path=gz_path(code)
        if not path.exists():
            coverage.append({"district_code":code,"district":district,"http_export_cached":False,
              "rows":None,"estate_count":None,"block_count":None,"complete_source_response":False,"source_error":"no validated cache"})
            continue
        raw,payload=load_cache(code); data=payload["data"]
        if not data:
            usable=payload.get("status") != "failed_empty_export"
            coverage.append({"district_code":code,"district":district,"http_export_cached":True,
              "rows":0,"estate_count":0,"block_count":0,"complete_source_response":usable,
              "source_error":payload.get("status","")}); continue
        snapshot_sha=hashlib.sha256(raw).hexdigest()
        d=pd.DataFrame(data); d["_district_code"]=code
        d["_district_export_complete"]=payload.get("status") != "failed_empty_export"
        d["_snapshot_sha256_normalized"]=snapshot_sha
        all_rows.append(d)
        coverage.append({"district_code":code,"district":district,"http_export_cached":True,
          "rows":len(d),"estate_count":d.estate_english_name.nunique(),
          "block_count":d[["estate_english_name","english_name_of_block"]].drop_duplicates().shape[0],
          "complete_source_response":True,"source_error":""})
        print("aggregate",code,len(d),flush=True)
    pd.DataFrame(coverage).to_csv(HERE/"ha_bulk_district_coverage.csv",index=False,encoding="utf-8-sig")
    if not all_rows: return
    d=pd.concat(all_rows,ignore_index=True,sort=False)
    # Determine if flat_number repeats across floors before selecting a unit key.
    basekey=["_district_code","estate_english_name","english_name_of_block","flat_number"]
    fullkey=basekey+["floor_number"]
    repeated=d[d.duplicated(basekey,keep=False)]
    same_floor_repeats=int(d.duplicated(fullkey).sum())
    # Floor is included because flat numbers may recur across levels.
    usable=d.dropna(subset=fullkey).copy()
    duplicate_full=int(usable.duplicated(fullkey).sum())
    units=usable.drop_duplicates(fullkey,keep="first")
    grouped=[]
    for (code,estate,block),g in units.groupby(["_district_code","estate_english_name","english_name_of_block"],dropna=False,sort=True):
        labels=sorted({str(x).strip() for x in g.floor_number.dropna() if str(x).strip()})
        area=pd.to_numeric(g.internal_floor_area,errors="coerce")
        est=g[["estate_map_latitude","estate_map_longitude"]].dropna().drop_duplicates()
        grouped.append({"district_code":code,"estate_english_name":estate,"estate_chinese_name":g.estate_chinese_name.dropna().iloc[0] if g.estate_chinese_name.notna().any() else None,
          "english_name_of_block":block,"chinese_name_of_block":g.chinese_name_of_block.dropna().iloc[0] if g.chinese_name_of_block.notna().any() else None,
          "unique_flat_rows":len(g),"distinct_floor_label_count":len(labels),"distinct_floor_labels":"|".join(labels),
          "sum_internal_floor_area_sqm":float(area.sum(min_count=1)) if area.notna().any() else None,
          "internal_area_missing_rows":int(area.isna().sum()),
          "estate_map_latitude":est.iloc[0,0] if len(est)==1 else None,"estate_map_longitude":est.iloc[0,1] if len(est)==1 else None,
          "estate_centroid_conflict":len(est)>1,"district_response_complete":bool(g["_district_export_complete"].all()),
          "normalized_snapshot_sha256":g["_snapshot_sha256_normalized"].iloc[0],
          "actual_total_storeys":None,"building_gfa_sqm":None,
          "scope_note":"Distinct unit floor labels and listed flat internal-area sum; not actual storeys or GFA."})
    out=pd.DataFrame(grouped)
    # Exact normalized block-name candidates only; not a confirmed spatial match.
    names=pq.ParquetFile(PARQUET).schema_arrow.names
    prefixes=["Building_ID_","LandsD_Building_Name_EN_","LandsD_Building_Name_TC_"]
    cols=[next(n for n in names if n.startswith(pre)) for pre in prefixes]
    comps=pd.read_parquet(PARQUET,columns=cols).drop_duplicates(cols[0]); idc,enc,tcc=cols
    comps["_en_norm"]=comps[enc].map(norm); comps["_tc_norm"]=comps[tcc].map(norm)
    counts=[]; ids=[]; match_status=[]; is_single_candidate=[]
    for row in out.itertuples(index=False):
        en=norm(row.english_name_of_block); tc=norm(row.chinese_name_of_block)
        hits=comps[(comps["_en_norm"]==en)|((tc!="")&(comps["_tc_norm"]==tc))]
        count=len(hits); counts.append(count); ids.append("|".join(hits[idc].astype(str).tolist()))
        match_status.append("no exact name candidate" if count==0 else ("single name candidate; verify address/geometry" if count==1 else "ambiguous multiple name candidates; verify address/geometry"))
        is_single_candidate.append(count==1)
    out["landsd_exact_name_candidate_count"]=counts; out["landsd_candidate_component_ids"]=ids
    out["landsd_name_candidate_status"]=match_status
    out["landsd_has_single_exact_name_candidate"]=is_single_candidate
    out=out.sort_values(["district_code","estate_english_name","english_name_of_block"],kind="stable")
    out.to_csv(HERE/"ha_all_district_block_evidence.csv",index=False,encoding="utf-8-sig")
    out.to_parquet(HERE/"ha_all_district_block_evidence.parquet",index=False)
    cov=pd.DataFrame(coverage)
    audit={"total_response_rows":len(d),"districts_with_cached_snapshot":int(cov.http_export_cached.sum()),
      "districts_with_validated_source_response":int(cov.complete_source_response.sum()),
      "districts_incomplete_or_failed":int((~cov.complete_source_response).sum()),
      "districts_expected":len(DISTRICTS),"units_missing_any_full_key_field":int(d[fullkey].isna().any(axis=1).sum()),
      "flat_number_reused_across_floor_or_level_key_rows":int(d.duplicated(basekey).sum()),
      "flat_number_reused_key_record_rows":len(repeated),
      "flat_number_only_key_is_not_used_for_final_dedup":"floor_number is included conservatively because the official dictionary does not specify that flat_number alone encodes level, even though no duplicate estate/block/flat keys occurred in these returned records.",
      "unique_estate_block_flat_keys_without_floor":int(d[basekey].drop_duplicates().shape[0]),
      "duplicate_rows_on_estate_block_flat_key":int(d.duplicated(basekey).sum()),
      "unique_composite_unit_keys_including_floor":int(usable[fullkey].drop_duplicates().shape[0]),
      "duplicate_rows_on_composite_unit_key_including_floor":duplicate_full,
      "deduped_unit_rows_used_for_aggregates":len(units),"block_rows":len(out),
      "floors_are_distinct_labels_not_storey_counts":True,"internal_area_sum_is_not_gfa":True,
      "complete_source_response_definition":"HTTP 200 valid JSON export fully cached; does not prove that HA source covers vacant/non-PRH/non-residential or building component inventory."}
    (HERE/"ha_bulk_aggregation_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot_directory',type=Path)
    parser.add_argument('building_parquet',type=Path)
    args=parser.parse_args()
    HERE,PARQUET=args.snapshot_directory,args.building_parquet
    manifest=pd.read_csv(HERE/'ha_bulk_download_manifest.csv',dtype=str,keep_default_na=False)
    for row in manifest.itertuples(index=False):
        raw,_=load_cache(row.district_code)
        if hashlib.sha256(raw).hexdigest()!=row.sha256_normalized_snapshot:
            raise ValueError('Normalized HA snapshot checksum mismatch: '+row.district_code)
    aggregate()
