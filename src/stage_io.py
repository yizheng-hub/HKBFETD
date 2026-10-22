"""Checked one-building-per-row interchange for the numbered stages."""
from pathlib import Path
import geopandas as gpd
import pandas as pd
import pyogrio
from data_quality import ID, ids, require_unique
from utils import sanitize_dataframe_for_geojson


def read_table(path):
    frame = pd.read_csv(path, low_memory=False, dtype={ID: 'string'})
    frame[ID] = ids(frame[ID])
    require_unique(frame)
    return frame


def attach_geometry(frame, geometry_path):
    geometry = pyogrio.read_dataframe(geometry_path, columns=[ID])
    geometry[ID] = ids(geometry[ID])
    require_unique(geometry)
    require_unique(frame)
    if set(frame[ID]) != set(geometry[ID]):
        raise ValueError('Geometry/attribute building coverage differs')
    return gpd.GeoDataFrame(frame.drop(columns='geometry', errors='ignore').merge(
        geometry, on=ID, validate='one_to_one'), geometry='geometry', crs=geometry.crs)


def save_buildings(frame, csv_path):
    require_unique(frame)
    frame = frame.sort_values(ID, kind='stable').reset_index(drop=True)
    safe = sanitize_dataframe_for_geojson(frame)
    safe.drop(columns='geometry', errors='ignore').to_csv(csv_path, index=False, encoding='utf-8-sig')
    if 'geometry' in safe:
        gpd.GeoDataFrame(safe, geometry='geometry', crs=frame.crs).to_file(Path(csv_path).with_suffix('.geojson'), driver='GeoJSON', AUTODETECT_JSON_STRINGS='NO')
    return frame
