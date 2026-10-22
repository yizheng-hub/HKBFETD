"""Freeze public CSDI Building attributes for a source-quality comparison.

This is an explicit acquisition command; it does not modify pipeline results.
No API key is needed for the public ArcGIS attribute layer.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import httpx
import pandas as pd
from ai_cache import file_hash

SERVICE = 'https://portal.csdi.gov.hk/server/rest/services/common/landsd_rcd_1637211194312_35158/MapServer/0'
FIELDS = ['OBJECTID','BuildingCSUID','BuildingID','GeoRefNo','BuildingBlockType','Status',
    'BuildingNameEN','BuildingNameTC','BaseHeight','TopHeight','Storeys','StoreysInBasement',
    'DateCreate','DateStamp','Shape_Area','Shape_Length']


def fetch(output):
    output.parent.mkdir(parents=True,exist_ok=True)
    pages=output.with_suffix('.pages')
    pages.mkdir(exist_ok=True)
    with httpx.Client(timeout=60,follow_redirects=True) as client:
        def query(parameters):
            for attempt in range(3):
                try:
                    time.sleep(1.5)  # Keep public-service traffic below 40 requests/minute.
                    response=client.get(SERVICE+'/query',params={'f':'json',**parameters})
                    if response.status_code in [401,403]:
                        raise PermissionError('Public source denied this request; acquisition stopped and completed pages preserved')
                    response.raise_for_status()
                    data=response.json()
                    if 'error' in data:
                        raise RuntimeError('Public source returned ArcGIS error '+str(data['error'].get('code')))
                    return data
                except (httpx.HTTPError,RuntimeError):
                    if attempt==2:raise
                    time.sleep(attempt+1)
        identity=query({'where':'1=1','returnIdsOnly':'true'})
        object_ids=set(identity['objectIds'])
        page_size=3000
        def page(offset):
            cached=pages/f'{offset:06d}.json'
            if cached.exists():
                return json.loads(cached.read_text(encoding='utf-8'))
            result=query({'where':'1=1','outFields':','.join(FIELDS),'returnGeometry':'false',
                'orderByFields':'OBJECTID ASC','resultOffset':offset,'resultRecordCount':page_size})
            rows=[feature['attributes'] for feature in result['features']]
            pending=cached.with_suffix('.pending')
            pending.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
            pending.replace(cached)
            return rows
        rows=[]
        for index,offset in enumerate(range(0,len(object_ids),page_size),start=1):
            rows.extend(page(offset))
            if index%10==0:print(f'[INFO] Frozen {len(rows):,} of {len(object_ids):,} source records',flush=True)
    frame=pd.DataFrame(rows).sort_values('OBJECTID')
    if frame.OBJECTID.duplicated().any() or set(frame.OBJECTID)!=object_ids:
        raise ValueError('Source changed during pagination or contains duplicate/missing IDs; snapshot rejected')
    if frame.BuildingCSUID.isna().any() or frame.BuildingCSUID.duplicated().any():
        raise ValueError('Source BuildingCSUID is not unique')
    pending=output.with_suffix('.pending')
    frame.to_csv(pending,index=False,encoding='utf-8-sig')
    pending.replace(output)
    height=pd.to_numeric(frame.TopHeight,errors='coerce')-pd.to_numeric(frame.BaseHeight,errors='coerce')
    metadata={'source':SERVICE,'provider':'Hong Kong Lands Department / CSDI Building',
        'retrieved_utc':datetime.now(timezone.utc).isoformat(),'rows':len(frame),'sha256':file_hash(output),
        'id_set_complete':True,'csuid_unique':True,'fields':FIELDS,
        'base_height_nonmissing':int(frame.BaseHeight.notna().sum()),'top_height_nonmissing':int(frame.TopHeight.notna().sum()),
        'positive_top_minus_base':int(height.gt(0).sum()),'reported_storeys_nonmissing':int(frame.Storeys.notna().sum()),
        'warning':'Top/base elevations and structural height must be distinguished from terrain-relative and architectural heights; validate component matching before integration.'}
    output.with_suffix('.metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[INFO] Public CSDI snapshot complete: '+json.dumps({k:v for k,v in metadata.items() if k not in ['fields','warning']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    fetch(args.output)
