# data_preprocessing.py


# -*- coding: utf-8 -*-
import os
import json
import sys
import warnings
import traceback
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
import pyrosm
import folium
from pyproj import Transformer
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import BallTree

import pyarrow.parquet as pq
from shapely import wkb

try:
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

from config import (
    BASE_DIR, DATA_DIR, INTERMEDIATE_DIR, OUTPUT_DIR, CTL_DIR,
    FILE_PATHS_LANDSD, BDBIAR_FILE_PATH, OSM_PBF_FILE_PATH,
    OZP_DATA_DIR, KEYWORDS_FILE,
    OFFICIAL_LIBRARY_PATH, OFFICIAL_LIB_BASE_PATH,
    ANATOMY_MAP_PATH, UNMATCHED_BD_MAP_PATH,
    MAX_MATCH_DISTANCE, TARGET_LON, TARGET_LAT, SEARCH_RADIUS_METERS,
    RUN_MICRO_ANATOMY, RUN_DEEP_VALIDATION, RUN_UNMATCHED_BD_INVESTIGATION,
    RUN_REVERSE_TRACEABILITY, TARGET_LANDSD_BSID_TO_INVESTIGATE,
    COMPACTNESS_EPSILON, AREA_THRESHOLD_VILLAGE, FLOOR_HEIGHT_ESTIMATE,
    MIN_FLOORS_FOR_ESTIMATE, OSM_ALL_PATH, BDBIAR_CACHE_PATH, AGGREGATED_GDF_PATH
)

from utils import (
    to_simplified_chinese, safe_str, init_keyword_tool,
    classify_text_by_keywords, classify_from_bdbiar_final,
    estimate_gfa_for_validation, load_and_fuse_ozp
)

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
tqdm.pandas()

def load_landsd_json(file_path, is_geo=False):
    try:
        if is_geo:
            gdf = gpd.read_file(file_path)
            gdf = gdf.set_crs("EPSG:2326", allow_override=True)
            return gdf
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            properties = [feature['properties'] for feature in data['features']]
            return pd.DataFrame(properties)
    except FileNotFoundError:
        return gpd.GeoDataFrame() if is_geo else pd.DataFrame()
    except Exception as e:
        return gpd.GeoDataFrame() if is_geo else pd.DataFrame()

def load_bdbiar_csv_as_gdf(file_path):
    try:
        df = pd.read_csv(file_path)
        df.dropna(subset=['LONGITUDE', 'LATITUDE'], inplace=True)
        gdf = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE),
            crs="EPSG:4326"
        )
        return gdf
    except FileNotFoundError:
        return gpd.GeoDataFrame()
    except Exception as e:
        return gpd.GeoDataFrame()


def load_overture_places(file_path):
    if not os.path.exists(file_path):
        return gpd.GeoDataFrame()
        
    try:
        table = pq.read_table(file_path)
        df_overture = table.to_pandas()
        
        from shapely import wkb
        df_overture['geometry'] = df_overture['geometry'].apply(lambda x: wkb.loads(x) if pd.notna(x) else None)
        gdf_overture = gpd.GeoDataFrame(df_overture, geometry='geometry', crs="EPSG:4326")
        
        gdf_overture = gdf_overture.to_crs("EPSG:2326")
        
        
        def extract_name(name_struct):
            if pd.isna(name_struct) or not name_struct: return ""
            if isinstance(name_struct, dict):
                return name_struct.get('primary', '')
            return str(name_struct)
            
        gdf_overture['name'] = gdf_overture['names'].apply(extract_name)
        
        def process_category(cat_struct):
            if pd.isna(cat_struct) or not cat_struct: return None, None
            
            main_cat = cat_struct.get('primary', '').lower() if isinstance(cat_struct, dict) else str(cat_struct).lower()
            
            retail_keywords = ['retail', 'shop', 'store', 'market', 'grocery', 'boutique', 'supermarket', 'convenience', 'mall']
            food_keywords = ['restaurant', 'cafe', 'food', 'bar', 'dining', 'bakery', 'coffee']
            office_keywords = ['office', 'corporate', 'company', 'agency']
            
            if any(k in main_cat for k in food_keywords):
                return 'restaurant', None
            elif any(k in main_cat for k in retail_keywords):
                return None, 'retail'
            elif any(k in main_cat for k in office_keywords):
                return None, 'office'
            elif 'hotel' in main_cat or 'accommodation' in main_cat:
                return 'hotel', None
            elif 'hospital' in main_cat or 'clinic' in main_cat:
                return 'hospital', None
            elif 'school' in main_cat or 'education' in main_cat:
                return 'school', None
                
            return main_cat, None
            
        cat_result = gdf_overture['categories'].apply(process_category)
        gdf_overture['amenity'] = [x[0] for x in cat_result]
        gdf_overture['shop'] = [x[1] for x in cat_result]
        
        gdf_overture['data_source'] = 'overture'
        gdf_overture['osmid'] = 'ovt_' + gdf_overture['id'].astype(str)
        
        cols_to_keep = ['osmid', 'name', 'amenity', 'shop', 'data_source', 'geometry']
        gdf_overture = gdf_overture[[c for c in cols_to_keep if c in gdf_overture.columns]]
        
        gdf_overture = gdf_overture[gdf_overture.geometry.is_valid & ~gdf_overture.geometry.is_empty]
        
        food_count = (gdf_overture['amenity'] == 'restaurant').sum()
        retail_count = (gdf_overture['shop'] == 'retail').sum()
        office_count = (gdf_overture['shop'] == 'office').sum()
        
        return gdf_overture
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return gpd.GeoDataFrame()


def load_all_data():
    
    dfs_landsd = {}
    dfs_landsd['BUILDING_STRUCTURE'] = load_landsd_json(FILE_PATHS_LANDSD['BUILDING_STRUCTURE'], is_geo=True)
    for name, path in FILE_PATHS_LANDSD.items():
        if name != 'BUILDING_STRUCTURE':
            dfs_landsd[name] = load_landsd_json(path, is_geo=False)
    
    gdf_bdbiar = load_bdbiar_csv_as_gdf(BDBIAR_FILE_PATH)
    
    
    OSM_ALL_CACHE = os.path.join(INTERMEDIATE_DIR, "step1_osm_features_all.geojson")
    
    use_cache = False
    if os.path.exists(OSM_ALL_CACHE):
        try:
            file_size = os.path.getsize(OSM_ALL_CACHE) / (1024 * 1024)  # MB
            if file_size > 1:
                use_cache = True
            else:
                pass
        except Exception as e:
            pass
    
    if use_cache:
        try:
            gdf_osm_all = gpd.read_file(OSM_ALL_CACHE)
            
            gdf_osm_buildings = gdf_osm_all[
                gdf_osm_all.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
            ].copy()
            gdf_osm_pois = gdf_osm_all[
                gdf_osm_all.geometry.geom_type.isin(['Point'])
            ].copy()
            
            return dfs_landsd, gdf_bdbiar, gdf_osm_buildings, gdf_osm_pois
        except Exception as e:
            traceback.print_exc()
    
    try:
        osm_reader = pyrosm.OSM(OSM_PBF_FILE_PATH)
        
        gdf_osm_buildings_raw = osm_reader.get_buildings()
        gdf_osm_buildings = gdf_osm_buildings_raw if gdf_osm_buildings_raw is not None else gpd.GeoDataFrame()
        if not gdf_osm_buildings.empty:
            pass
        else:
            pass
        
        gdf_osm_pois_raw = osm_reader.get_pois()
        gdf_osm_pois = gdf_osm_pois_raw if gdf_osm_pois_raw is not None else gpd.GeoDataFrame()
        if not gdf_osm_pois.empty:
            pass
        else:
            pass
        
        if not gdf_osm_buildings.empty or not gdf_osm_pois.empty:
            gdf_osm_all = pd.concat([gdf_osm_buildings, gdf_osm_pois], ignore_index=True)
            if 'id' in gdf_osm_all.columns:
                gdf_osm_all.rename(columns={'id': 'osmid'}, inplace=True)
            gdf_osm_all.to_file(OSM_ALL_CACHE, driver='GeoJSON')
        
        
    except FileNotFoundError:
        gdf_osm_buildings = gpd.GeoDataFrame()
        gdf_osm_pois = gpd.GeoDataFrame()
    except Exception as e:
        traceback.print_exc()
        gdf_osm_buildings = gpd.GeoDataFrame()
        gdf_osm_pois = gpd.GeoDataFrame()
    
    return dfs_landsd, gdf_bdbiar, gdf_osm_buildings, gdf_osm_pois

def inspect_data(dfs_landsd, gdf_bdbiar, gdf_osm_buildings, gdf_osm_pois):
    
    if 'BUILDING_STRUCTURE' in dfs_landsd and not dfs_landsd['BUILDING_STRUCTURE'].empty:
        print(dfs_landsd['BUILDING_STRUCTURE'].head(2))
    else:
        pass
    
    if 'BUILDING_NAME' in dfs_landsd and not dfs_landsd['BUILDING_NAME'].empty:
        print(dfs_landsd['BUILDING_NAME'].head(2))
    else:
        pass
    
    if not gdf_bdbiar.empty:
        print(gdf_bdbiar[['ADDRESS_C', 'NSEARCH5_C', 'geometry']].head(2))
    else:
        pass
    
    if not gdf_osm_buildings.empty:
        osm_building_cols = ['name', 'building', 'amenity', 'shop', 'geometry']
        valid_cols = [col for col in osm_building_cols if col in gdf_osm_buildings.columns]
        print(gdf_osm_buildings[valid_cols].head(2))
    else:
        pass
        
    if not gdf_osm_pois.empty:
        osm_poi_cols = ['name', 'amenity', 'shop', 'office', 'cuisine', 'geometry']
        valid_cols = [col for col in osm_poi_cols if col in gdf_osm_pois.columns]
        print(gdf_osm_pois[valid_cols].head(2))
    else:
        pass

def validate_data_integrity(gdf):
    issues = []
    
    invalid_geoms = gdf[~gdf.geometry.is_valid]
    if len(invalid_geoms) > 0:
        issues.append(f"Detected {len(invalid_geoms)} invalid geometries")
    
    empty_geoms = gdf[gdf.geometry.is_empty]
    if len(empty_geoms) > 0:
        issues.append(f"Detected {len(empty_geoms)} empty geometries")
    
    duplicate_ids = gdf['BUILDINGSTRUCTUREID'].duplicated().sum()
    if duplicate_ids > 0:
        issues.append(f"Detected {duplicate_ids} duplicated BUILDINGSTRUCTUREID values")
    
    if not gdf.empty and 'geometry' in gdf.columns:
        areas = gdf.geometry.area
        area_stats = areas.describe()
        
        tiny_areas = (areas < 1).sum()
        if tiny_areas > 0:
            issues.append(f"Detected {tiny_areas} geometries with area smaller than 1 m2")
    
    if not issues:
        pass
    
    return issues

def validate_and_fix_data_integrity(gdf):
    issues = []
    
    original_crs = gdf.crs
    
    
    def clean_geometry(geom):
        if geom is None or geom.is_empty:
            return geom
        
        try:
            if hasattr(geom, 'exterior'):
                coords = list(geom.exterior.coords)
                for coord in coords:
                    if any(np.isnan(c) or np.isinf(c) for c in coord):
                        return None
            elif hasattr(geom, 'x') and hasattr(geom, 'y'):
                if np.isnan(geom.x) or np.isinf(geom.x) or np.isnan(geom.y) or np.isinf(geom.y):
                    return None
            
            return geom
        except Exception as e:
            return None
    
    original_count = len(gdf)
    gdf_fixed = gdf.copy()
    
    to_drop = []
    for idx, row in tqdm(gdf_fixed.iterrows(), total=len(gdf_fixed), desc="Cleaning geometries"):
        geom = row.geometry
        if geom is None or geom.is_empty:
            to_drop.append(idx)
            continue
            
        cleaned_geom = clean_geometry(geom)
        if cleaned_geom is None:
            to_drop.append(idx)
        else:
            gdf_fixed.at[idx, 'geometry'] = cleaned_geom
    
    if to_drop:
        gdf_fixed = gdf_fixed.drop(to_drop)
    
    from shapely.validation import make_valid
    
    invalid_mask = ~gdf_fixed.geometry.is_valid
    invalid_count = invalid_mask.sum()
    
    if invalid_count > 0:
        
        def safe_make_valid(geom):
            try:
                if geom is None or geom.is_empty:
                    return geom
                try:
                    fixed = geom.buffer(0)
                    if fixed.is_valid:
                        return fixed
                except:
                    pass
                
                try:
                    return make_valid(geom)
                except Exception as e:
                    return None
            except Exception as e:
                return None
        
        gdf_fixed.loc[invalid_mask, 'geometry'] = gdf_fixed.loc[invalid_mask, 'geometry'].apply(safe_make_valid)
        
        still_invalid = ~gdf_fixed.geometry.is_valid
        still_invalid_count = still_invalid.sum()
        if still_invalid_count > 0:
            gdf_fixed = gdf_fixed[~still_invalid].copy()
    
    if original_crs is not None:
        gdf_fixed.crs = original_crs
    else:
        gdf_fixed.crs = "EPSG:2326"
    
    final_count = len(gdf_fixed)
    removed_count = original_count - final_count
    if removed_count > 0:
        issues.append(f"Removed {removed_count} invalid geometries")
    
    
    return gdf_fixed, issues

def print_memory_usage(label=""):
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        return mem_mb
    except ImportError:
        return None
    except Exception as e:
        return None

def cleanup_memory(*args):
    import gc
    for arg in args:
        if arg is not None:
            del arg
    gc.collect()

def improve_matching_with_address_similarity(gdf_landsd, gdf_bd, max_distance=30, landsd_name_col='BUILDINGNAMETC', bd_address_col='ADDRESS_C'):
    if not FUZZY_AVAILABLE:
        return None
    
    
    if landsd_name_col not in gdf_landsd.columns:
        return None
    
    if bd_address_col not in gdf_bd.columns:
        return None
    
    def calculate_similarity(row):
        try:
            name1 = str(row.get(landsd_name_col, ''))
            name2 = str(row.get(bd_address_col, ''))
            if pd.isna(name1) or pd.isna(name2):
                return 0
            return fuzz.ratio(name1, name2)
        except:
            return 0
    
    if 'distance_to_bdbiar' in gdf_landsd.columns:
        candidates = gdf_landsd.copy()
        candidates['distance'] = candidates['distance_to_bdbiar']
    else:
        candidates = gpd.sjoin_nearest(
            gdf_landsd, gdf_bd, 
            how='left', 
            max_distance=max_distance,
            distance_col='distance'
        )
    
    candidates['address_similarity'] = candidates.apply(calculate_similarity, axis=1)
    
    candidates['match_score'] = (
        (1 - candidates['distance'].fillna(max_distance) / max_distance) * 0.6 +
        (candidates['address_similarity'] / 100) * 0.4
    )
    
    best_matches = candidates.sort_values(
        ['BUILDINGSTRUCTUREID', 'match_score'], 
        ascending=[True, False]
    ).groupby('BUILDINGSTRUCTUREID').first().reset_index()
    
    return best_matches

def build_official_library(dfs_landsd, gdf_bdbiar, use_address_similarity=False):
    
    if 'BUILDING_STRUCTURE' not in dfs_landsd or dfs_landsd['BUILDING_STRUCTURE'].empty:
        raise ValueError("Failed to load base building geometry (BUILDING_STRUCTURE).")
    
    gdf_base = dfs_landsd['BUILDING_STRUCTURE'].copy()
    
    if 'BUILDING_NAME' in dfs_landsd and not dfs_landsd['BUILDING_NAME'].empty:
        df_name = dfs_landsd['BUILDING_NAME'].sort_values('NAMESTATUS').drop_duplicates('BUILDINGSTRUCTUREID')
        gdf_base = gdf_base.merge(
            df_name[['BUILDINGSTRUCTUREID', 'BUILDINGNAMEEN', 'BUILDINGNAMETC']], 
            on='BUILDINGSTRUCTUREID', how='left'
        )
    
    if 'OP_BUILDING_STRUCTURE' in dfs_landsd and 'OCCUPATION_PERMIT' in dfs_landsd and not dfs_landsd['OP_BUILDING_STRUCTURE'].empty:
        df_op_permit = dfs_landsd['OCCUPATION_PERMIT'].copy()
        df_op_permit['DOMESTICGFA'] = pd.to_numeric(df_op_permit['DOMESTICGFA'], errors='coerce')
        df_op_permit['NONDOMESTICGFA'] = pd.to_numeric(df_op_permit['NONDOMESTICGFA'], errors='coerce')
        df_op_full = dfs_landsd['OP_BUILDING_STRUCTURE'].merge(df_op_permit, on='OPNO', how='left')
        df_gfa_agg = df_op_full.groupby('BUILDINGSTRUCTUREID').agg(
            GFA_DOMESTIC_SUM=('DOMESTICGFA', 'sum'), 
            GFA_NONDOMESTIC_SUM=('NONDOMESTICGFA', 'sum')
        ).reset_index()
        gdf_base = gdf_base.merge(df_gfa_agg, on='BUILDINGSTRUCTUREID', how='left')
    
    if 'BUILDING_INFO' in dfs_landsd and not dfs_landsd['BUILDING_INFO'].empty:
        df_info_agg = dfs_landsd['BUILDING_INFO'].groupby('BUILDINGSTRUCTUREID')['INFODESCRIPTION'].apply(
            lambda x: '; '.join(x.dropna().unique())
        ).reset_index().rename(columns={'INFODESCRIPTION': 'ALL_INFO_DESC'})
        gdf_base = gdf_base.merge(df_info_agg, on='BUILDINGSTRUCTUREID', how='left')
    
    
    
    if 'gdf_bdbiar' not in locals() or gdf_bdbiar.empty:
        gdf_official_library = gdf_base.copy()
        gdf_official_library['distance_to_bdbiar'] = np.nan
        gdf_official_library['BDBIAR_OBJECTID'] = np.nan
    else:
        gdf_bdbiar_reprojected = gdf_bdbiar.to_crs(gdf_base.crs)
        gdf_bdbiar_reprojected.rename(columns={'OBJECTID': 'BDBIAR_OBJECTID'}, inplace=True)
        
        gdf_landsd_towers = gdf_base[gdf_base['BUILDINGSTRUCTURETYPE'] == 'T'].copy()
        gdf_landsd_podiums = gdf_base[gdf_base['BUILDINGSTRUCTURETYPE'] == 'P'].copy()
        gdf_bd_towers = gdf_bdbiar_reprojected[gdf_bdbiar_reprojected['NSEARCH4_C'].str.contains('座', na=False)].copy()
        gdf_bd_podiums = gdf_bdbiar_reprojected[gdf_bdbiar_reprojected['NSEARCH4_C'].str.contains('平台', na=False)].copy()
        
        joined_towers = gpd.sjoin_nearest(
            gdf_landsd_towers, gdf_bd_towers, how='left', 
            max_distance=MAX_MATCH_DISTANCE, distance_col='d_t'
        )
        joined_podiums = gpd.sjoin_nearest(
            gdf_landsd_podiums, gdf_bd_podiums, how='left', 
            max_distance=MAX_MATCH_DISTANCE, distance_col='d_p'
        )
        
        unique_towers = joined_towers.sort_values(['BUILDINGSTRUCTUREID', 'd_t']).groupby('BUILDINGSTRUCTUREID').first().reset_index()
        unique_podiums = joined_podiums.sort_values(['BUILDINGSTRUCTUREID', 'd_p']).groupby('BUILDINGSTRUCTUREID').first().reset_index()
        matched_in_step1 = pd.concat([
            unique_towers.dropna(subset=['BDBIAR_OBJECTID']), 
            unique_podiums.dropna(subset=['BDBIAR_OBJECTID'])
        ])
        matched_landsd_ids_step1 = matched_in_step1['BUILDINGSTRUCTUREID'].unique()
        matched_bd_ids_step1 = matched_in_step1['BDBIAR_OBJECTID'].unique()
        
        unmatched_landsd = gdf_base[~gdf_base['BUILDINGSTRUCTUREID'].isin(matched_landsd_ids_step1)]
        unmatched_bd = gdf_bdbiar_reprojected[~gdf_bdbiar_reprojected['BDBIAR_OBJECTID'].isin(matched_bd_ids_step1)]
        
        if not unmatched_landsd.empty and not unmatched_bd.empty:
            joined_others = gpd.sjoin_nearest(
                unmatched_landsd, unmatched_bd, how='left', 
                max_distance=MAX_MATCH_DISTANCE, distance_col='d_o'
            )
            unique_others = joined_others.sort_values(['BUILDINGSTRUCTUREID', 'd_o']).groupby('BUILDINGSTRUCTUREID').first().reset_index()
        else:
            unique_others = unmatched_landsd.copy()
            unique_others['d_o'] = np.nan
        
        
        if 'd_t' in matched_in_step1.columns:
            matched_in_step1.rename(columns={'d_t': 'distance_to_bdbiar'}, inplace=True)
        elif 'd_p' in matched_in_step1.columns:
            matched_in_step1.rename(columns={'d_p': 'distance_to_bdbiar'}, inplace=True)
        
        if 'd_o' in unique_others.columns:
            unique_others.rename(columns={'d_o': 'distance_to_bdbiar'}, inplace=True)
        
        gdf_official_library = pd.concat([matched_in_step1, unique_others], ignore_index=True)
        
        if 'distance_to_bdbiar' not in gdf_official_library.columns:
            gdf_official_library['distance_to_bdbiar'] = np.nan
        
        if 'index_right' in gdf_official_library.columns:
            gdf_official_library.drop(columns=['index_right'], inplace=True, errors='ignore')
        
        column_renames = {
            'ADDRESS_C': 'BDBIAR_ADDRESS_C',
            'NSEARCH5_C': 'BDBIAR_CLASS',
            'NSEARCH3_C': 'BDBIAR_NSEARCH3_C',
            'NSEARCH4_C': 'BDBIAR_NSEARCH4_C'
        }
        
        for old_col, new_col in column_renames.items():
            if old_col in gdf_official_library.columns:
                gdf_official_library.rename(columns={old_col: new_col}, inplace=True)
        
        if 'BDBIAR_NSEARCH3_C' in gdf_official_library.columns:
            gdf_official_library['BDBIAR_AGE'] = pd.to_datetime(
                gdf_official_library['BDBIAR_NSEARCH3_C'], errors='coerce'
            ).dt.year
        
    
    if use_address_similarity and not gdf_bdbiar.empty:
        
        matched_records = gdf_official_library[gdf_official_library['BDBIAR_OBJECTID'].notna()].copy()
        
        if not matched_records.empty:
            improved_matches = improve_matching_with_address_similarity(
                matched_records, 
                gdf_bdbiar_reprojected,
                max_distance=MAX_MATCH_DISTANCE
            )
            
            if improved_matches is not None:
                pass

    
    display_cols = ['BUILDINGSTRUCTUREID', 'BUILDINGNAMETC', 'GFA_DOMESTIC_SUM', 'BDBIAR_CLASS', 'BDBIAR_AGE']
    available_cols = [col for col in display_cols if col in gdf_official_library.columns]
    if available_cols:
        print(gdf_official_library[available_cols].head().to_markdown())
    
    gdf_official_library.to_file(OFFICIAL_LIBRARY_PATH, driver='GeoJSON')
    
    return gdf_official_library

def micro_anatomy(gdf_official_library):
    if not RUN_MICRO_ANATOMY:
        return
    
    
    if gdf_official_library.crs is None:
        gdf_official_library = gdf_official_library.set_crs("EPSG:2326")
    
    try:
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2326", always_xy=True)
        center_x, center_y = transformer.transform(TARGET_LON, TARGET_LAT)
        aoi_polygon = gpd.GeoDataFrame(
            geometry=[Point(center_x, center_y).buffer(SEARCH_RADIUS_METERS)], 
            crs="EPSG:2326"
        )
        
        gdf_aoi = gpd.sjoin(gdf_official_library, aoi_polygon, how='inner', predicate='intersects')
        
        gdf_podiums_aoi = gdf_aoi[gdf_aoi['BUILDINGSTRUCTURETYPE'] == 'P'].copy()
        gdf_towers_aoi = gdf_aoi[gdf_aoi['BUILDINGSTRUCTURETYPE'] == 'T'].copy()
        gdf_others_aoi = gdf_aoi[~gdf_aoi['BUILDINGSTRUCTURETYPE'].isin(['P', 'T'])].copy()


    except Exception as e:
        traceback.print_exc()
        return

    def add_layer_to_map(gdf_to_plot, color, layer_name, map_object):
        if gdf_to_plot.empty:
            return
        
        if gdf_to_plot.crs is None:
            gdf_to_plot = gdf_to_plot.set_crs("EPSG:2326")
        
        try:
            gdf_wgs84 = gdf_to_plot.to_crs("EPSG:4326")
        except Exception as e:
            return
        
        feature_group = folium.FeatureGroup(name=layer_name, show=True)
        
        for _, row in gdf_wgs84.iterrows():
            popup_html = "<h4>建筑属性详情</h4><hr><table>"
            for col in ['BUILDINGSTRUCTUREID', 'BUILDINGNAMETC', 'BUILDINGSTRUCTURETYPE', 'CATEGORY', 'BDBIAR_CLASS', 'BDBIAR_AGE']:
                if col in row:
                    popup_html += f"<tr><td style='font-weight:bold; padding-right:10px;'>{col}:</td><td>{safe_str(row.get(col))}</td></tr>"
            popup_html += f"<tr><td style='font-weight:bold; padding-right:10px;'>Area (sqm):</td><td>{row.geometry.area:.2f}</td></tr>"
            popup_html += "</table>"
            
            folium.GeoJson(
                row.geometry,
                style_function=lambda x, c=color: {'color': c, 'weight': 1.5, 'fillOpacity': 0.5, 'fillColor': c},
                tooltip=f"<b>BSID: {row['BUILDINGSTRUCTUREID']}</b><br>Type: {row.get('BUILDINGSTRUCTURETYPE', 'N/A')}",
                popup=folium.Popup(popup_html, max_width=450)
            ).add_to(feature_group)
        
        feature_group.add_to(map_object)

    
    m_anatomy = folium.Map(location=[TARGET_LAT, TARGET_LON], zoom_start=17, tiles="CartoDB positron")
    
    add_layer_to_map(gdf_others_aoi, 'gray', '其他类型', m_anatomy)
    add_layer_to_map(gdf_podiums_aoi, 'blue', '平台 (Podiums)', m_anatomy)
    add_layer_to_map(gdf_towers_aoi, 'red', '塔楼 (Towers)', m_anatomy)
    
    folium.TileLayer(
        'https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', 
        attr='Google', name='Google Satellite', overlay=True
    ).add_to(m_anatomy)
    
    folium.LayerControl(collapsed=False).add_to(m_anatomy)
    m_anatomy.save(ANATOMY_MAP_PATH)
    

def estimate_gfa_for_validation_wrapper(row):
    from config import FLOOR_HEIGHT_ESTIMATE, MIN_FLOORS_FOR_ESTIMATE
    config = {
        'FLOOR_HEIGHT_ESTIMATE': FLOOR_HEIGHT_ESTIMATE,
        'MIN_FLOORS_FOR_ESTIMATE': MIN_FLOORS_FOR_ESTIMATE
    }
    return estimate_gfa_for_validation(row, config)

def check_distance_data(df_val):
    total_matched = df_val['Is_Matched_BD'].sum()
    
    if 'distance_to_bdbiar' not in df_val.columns:
        for col in ['d_t', 'd_p', 'd_o', 'distance']:
            if col in df_val.columns:
                pass
        return
    
    has_distance = df_val['distance_to_bdbiar'].notna().sum()
    missing_distance = total_matched - has_distance
    
    
    zero_distance = (df_val['distance_to_bdbiar'] == 0).sum()
    if zero_distance > 0:
        pass
    
    if missing_distance > 0:
        missing_samples = df_val[df_val['Is_Matched_BD'] & df_val['distance_to_bdbiar'].isna()].head(5)
        if len(missing_samples) > 0:
            for idx, row in missing_samples.iterrows():
                print(f"    - BSID: {row['BUILDINGSTRUCTUREID']}, BDBIAR_ID: {row['BDBIAR_OBJECTID']}, " +
                      f"类型: {row.get('BUILDINGSTRUCTURETYPE', 'N/A')}")
        else:
            pass

def fix_missing_distance_data(gdf_official_library, gdf_bdbiar):
    
    if gdf_official_library.crs is None:
        gdf_official_library = gdf_official_library.set_crs("EPSG:2326")
    
    
    matched_but_missing = gdf_official_library[
        gdf_official_library['BDBIAR_OBJECTID'].notna() & 
        (gdf_official_library['distance_to_bdbiar'].isna() | (gdf_official_library['distance_to_bdbiar'] == 0))
    ].copy()
    
    if len(matched_but_missing) == 0:
        return gdf_official_library
    
    
    try:
        gdf_bdbiar_reprojected = gdf_bdbiar.to_crs(gdf_official_library.crs)
        gdf_bdbiar_reprojected.rename(columns={'OBJECTID': 'BDBIAR_OBJECTID'}, inplace=True)
        
        
        bd_ids_to_fix = matched_but_missing['BDBIAR_OBJECTID'].unique()
        bd_records_to_fix = gdf_bdbiar_reprojected[gdf_bdbiar_reprojected['BDBIAR_OBJECTID'].isin(bd_ids_to_fix)]
        
        if len(bd_records_to_fix) == 0:
            return gdf_official_library
        
        fixed_count = 0
        for idx, row in tqdm(matched_but_missing.iterrows(), total=len(matched_but_missing), desc="Repairing distance attributes"):
            bd_record = bd_records_to_fix[bd_records_to_fix['BDBIAR_OBJECTID'] == row['BDBIAR_OBJECTID']]
            if len(bd_record) > 0:
                bd_geom = bd_record.iloc[0].geometry
                distance = row.geometry.distance(bd_geom)
                gdf_official_library.at[idx, 'distance_to_bdbiar'] = distance
                fixed_count += 1
        
        
        still_missing = gdf_official_library[
            gdf_official_library['BDBIAR_OBJECTID'].notna() & 
            (gdf_official_library['distance_to_bdbiar'].isna() | (gdf_official_library['distance_to_bdbiar'] == 0))
        ]
        if len(still_missing) > 0:
            pass
        else:
            pass
        
    except Exception as e:
        traceback.print_exc()
    
    return gdf_official_library

def deep_validation(gdf_official_library):
    if not RUN_DEEP_VALIDATION:
        return
    
    
    df_val = gdf_official_library.copy()
    
    bdbiar_col = None
    if 'BDBIAR_OBJECTID' in df_val.columns:
        bdbiar_col = 'BDBIAR_OBJECTID'
    elif 'BDBIAR_O' in df_val.columns:
        bdbiar_col = 'BDBIAR_O'
    
    if bdbiar_col:
        df_val['Is_Matched_BD'] = df_val[bdbiar_col].notna()
    else:
        if 'BDBIAR_CLASS' in df_val.columns:
            df_val['Is_Matched_BD'] = df_val['BDBIAR_CLASS'].notna()
        else:
            df_val['Is_Matched_BD'] = False
    
    df_val['Val_GFA'] = df_val.apply(estimate_gfa_for_validation_wrapper, axis=1)
    
    total_gfa = df_val['Val_GFA'].sum()
    matched_gfa = df_val[df_val['Is_Matched_BD']]['Val_GFA'].sum()
    total_count = len(df_val)
    matched_count = df_val['Is_Matched_BD'].sum()
    coverage_ratio = matched_gfa / total_gfa if total_gfa > 0 else 0
    count_ratio = matched_count / total_count if total_count > 0 else 0
    
    
    if coverage_ratio > 0.80:
        pass
    else:
        pass
    
    df_val['Footprint_Area'] = df_val.geometry.area
    plot_data = df_val[df_val['Footprint_Area'] <= 2000].copy()
    
    try:
        plt.figure(figsize=(12, 6))
        sns.histplot(
            data=plot_data, 
            x='Footprint_Area', 
            hue='Is_Matched_BD', 
            element="step", 
            stat="count", 
            common_norm=False,
            bins=100,
            palette={True: "blue", False: "red"},
            log_scale=(False, True) 
        )
        
        plt.title('Distribution of Building Footprint Area (Log Scale)\nMatched BD (Core) vs. Unmatched (Tail)')
        plt.xlabel('Footprint Area (sq. meters)')
        plt.ylabel('Count of Buildings (Log Scale)')
        plt.legend(title='Source', labels=['Unmatched (LandsD Only)', 'Matched (BD+LandsD)'])
        plt.grid(True, which="both", ls="--", alpha=0.3)
        
        from config import AREA_THRESHOLD_VILLAGE
        village_house_area = AREA_THRESHOLD_VILLAGE
        plt.axvspan(village_house_area - 5, village_house_area + 5, color='orange', alpha=0.3, label='Typical Village House')
        plt.text(village_house_area, 100, "Village House\nTrap", color='darkorange', fontweight='bold', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "footprint_distribution.png"), dpi=150)
        plt.close()
    except Exception as e:
        pass
    

    check_distance_data(df_val)

    dist_col = 'distance_to_bdbiar'
    
    if dist_col in df_val.columns:
        matched_d = df_val[df_val['Is_Matched_BD']][dist_col]
        if matched_d.notna().any():
            matched_d_filtered = matched_d[matched_d > 0.001]
            
            
        else:
            pass
    else:
        pass


def investigate_unmatched_bd(gdf_official_library, gdf_bdbiar):
    if not RUN_UNMATCHED_BD_INVESTIGATION:
        return
    
    
    
    try:
        matched_bdbiar_ids = gdf_official_library['BDBIAR_OBJECTID'].dropna().unique()
        
        gdf_bd_unmatched = gdf_bdbiar[~gdf_bdbiar['OBJECTID'].isin(matched_bdbiar_ids)].copy()
        gdf_bd_unmatched.rename(columns={'OBJECTID': 'BDBIAR_OBJECTID'}, inplace=True)
        
        num_unmatched = len(gdf_bd_unmatched)
        
    except Exception as e:
        traceback.print_exc()
        return
    
    if gdf_bd_unmatched.empty:
        return
    
    
    plot_gdf = gdf_bd_unmatched.copy()
    plot_gdf_wgs84 = plot_gdf.to_crs("EPSG:4326")
    center_lat = plot_gdf_wgs84.geometry.y.mean()
    center_lon = plot_gdf_wgs84.geometry.x.mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")
    folium.TileLayer(
        'https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', 
        attr='Google', name='Google Satellite', overlay=True
    ).add_to(m)
    
    feature_group = folium.FeatureGroup(name=f"未能匹配的BD记录 ({len(plot_gdf)})")
    
    for _, row in tqdm(plot_gdf_wgs84.iterrows(), total=len(plot_gdf_wgs84), desc="Rendering map"):
        geom = row.geometry
        if geom and not geom.is_empty:
            popup_html = f"""
            <h4>未能匹配的BD记录</h4><hr>
            <b>BDBIAR_OBJECTID:</b> {row['BDBIAR_OBJECTID']}<br><hr>
            <b>地址:</b> {safe_str(row.get('ADDRESS_C'))}<br>
            <b>官方分类:</b> {safe_str(row.get('NSEARCH5_C'))}<br>
            <b>结构类型:</b> {safe_str(row.get('NSEARCH4_C'))}<br>
            <b>入伙年份:</b> {pd.to_datetime(row.get('NSEARCH3_C')).year if pd.notna(row.get('NSEARCH3_C')) else 'N/A'}
            """
            tooltip_text = f"ID: {row['BDBIAR_OBJECTID']} | {safe_str(row.get('ADDRESS_C'))}"
            
            folium.CircleMarker(
                location=[geom.y, geom.x], radius=5, color='orange', fill=True,
                fill_color='#ff7f0e', fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=450), 
                tooltip=tooltip_text
            ).add_to(feature_group)
    
    feature_group.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(UNMATCHED_BD_MAP_PATH)
    

def reverse_traceability(gdf_official_library, gdf_bdbiar):
    if not RUN_REVERSE_TRACEABILITY:
        return
    
    
    target_landsd_record = gdf_official_library[
        gdf_official_library['BUILDINGSTRUCTUREID'] == TARGET_LANDSD_BSID_TO_INVESTIGATE
    ]
    
    if target_landsd_record.empty:
        return
    
    target_landsd_record = target_landsd_record.iloc[0]
    
    matched_bd_id = target_landsd_record['BDBIAR_OBJECTID']
    
    if 'distance_to_bdbiar' in target_landsd_record:
        match_distance = target_landsd_record['distance_to_bdbiar']
    else:
        possible_dist_cols = ['d_t', 'd_p', 'd_o', 'distance']
        match_distance = None
        for col in possible_dist_cols:
            if col in target_landsd_record:
                match_distance = target_landsd_record[col]
                break
    
    
    if pd.notna(matched_bd_id):
        
        matched_bd_info = gdf_bdbiar[gdf_bdbiar['OBJECTID'] == int(matched_bd_id)]
        
        if not matched_bd_info.empty:
            matched_bd_row = matched_bd_info.iloc[0]
            info_cols = ['OBJECTID', 'ADDRESS_C', 'NSEARCH4_C', 'NSEARCH5_C']
            available_cols = [col for col in info_cols if col in matched_bd_row]
            print(matched_bd_row[available_cols])
        else:
            pass
        
        expected_bd_record = gdf_bdbiar[gdf_bdbiar['ADDRESS_C'].str.contains('啟岸', na=False)]
        if not expected_bd_record.empty:
            expected_bd_row = expected_bd_record.iloc[0]
            info_cols = ['OBJECTID', 'ADDRESS_C', 'NSEARCH4_C', 'NSEARCH5_C']
            available_cols = [col for col in info_cols if col in expected_bd_row]
            print(expected_bd_row[available_cols])
            
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:2326", always_xy=True)
            expected_lon = expected_bd_row.geometry.x
            expected_lat = expected_bd_row.geometry.y
            expected_point_hk80 = Point(transformer.transform(expected_lon, expected_lat))
            
            actual_distance = target_landsd_record.geometry.distance(expected_point_hk80)
            
            if actual_distance > MAX_MATCH_DISTANCE:
                pass
            else:
                pass
    else:
        pass

def check_data_preprocessing_dependencies():
    required_files = []
    
    for name, path in FILE_PATHS_LANDSD.items():
        if not os.path.exists(path):
            required_files.append(path)
    
    if not os.path.exists(BDBIAR_FILE_PATH):
        required_files.append(BDBIAR_FILE_PATH)
    
    if not os.path.exists(OSM_PBF_FILE_PATH):
        required_files.append(OSM_PBF_FILE_PATH)
    
    if not os.path.exists(KEYWORDS_FILE):
        required_files.append(KEYWORDS_FILE)
    
    if required_files:
        for f in required_files:
            print(f"  - {os.path.basename(f)}")
        return False
    
    return True

def spatial_join_with_index(gdf1, gdf2, **kwargs):
    if 'sindex' not in gdf1.attrs:
        gdf1.attrs['sindex'] = gdf1.sindex
    
    return gpd.sjoin(gdf1, gdf2, **kwargs)

def main():
    
    if not check_data_preprocessing_dependencies():
        return False
    
    print_memory_usage("Start")
    dfs_landsd, gdf_bdbiar, gdf_osm_buildings, gdf_osm_pois = load_all_data()
    print_memory_usage("After loading data")
    
    inspect_data(dfs_landsd, gdf_bdbiar, gdf_osm_buildings, gdf_osm_pois)
    
    gdf_official_library = build_official_library(dfs_landsd, gdf_bdbiar)
    print_memory_usage("After building official library")

    import glob
    from shapely.geometry import Polygon
    
    def load_ozp_data(ozp_dir):
        features = []
        json_files = glob.glob(os.path.join(ozp_dir, "*.json"))
        if not json_files:
            return gpd.GeoDataFrame()
        
        for f in tqdm(json_files, desc="Parsing OZP files"):
            with open(f, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    for feat in data.get('features', []):
                        attr = feat.get('attributes', {})
                        geom = feat.get('geometry', {})
                        if 'rings' in geom and len(geom['rings']) > 0:
                            poly = Polygon(geom['rings'][0])
                            features.append({
                                'geometry': poly, 
                                'OZP_ZONE_LABEL': attr.get('ZONE_LABEL', ''),
                                'OZP_DESC_ENG': attr.get('DESC_ENG', '')
                            })
                except Exception as e:
                    pass
        return gpd.GeoDataFrame(features, crs="EPSG:2326") if features else gpd.GeoDataFrame()

    gdf_ozp = load_ozp_data(OZP_DATA_DIR)
    if not gdf_ozp.empty:
        gdf_centroids = gdf_official_library.copy()
        gdf_centroids['geometry'] = gdf_centroids.geometry.centroid
        
        joined = gpd.sjoin(gdf_centroids, gdf_ozp[['geometry', 'OZP_ZONE_LABEL', 'OZP_DESC_ENG']], how='left', predicate='intersects')
        
        joined = joined[~joined.index.duplicated(keep='first')]
        
        gdf_official_library['OZP_ZONE_LABEL'] = joined['OZP_ZONE_LABEL']
        gdf_official_library['OZP_DESC_ENG'] = joined['OZP_DESC_ENG']
        
        matched_count = gdf_official_library['OZP_ZONE_LABEL'].notna().sum()
    else:
        pass

    
    objects_to_clean = []
    if 'gdf_osm_buildings' in locals():
        objects_to_clean.append(gdf_osm_buildings)
    if 'gdf_osm_pois' in locals():
        objects_to_clean.append(gdf_osm_pois)
    
    for obj in objects_to_clean:
        if obj is not None:
            del obj
    
    import gc
    collected = gc.collect()

    if 'dfs_landsd' in locals():
        for key in list(dfs_landsd.keys()):
            if key != 'BUILDING_STRUCTURE':
                del dfs_landsd[key]
        collected = gc.collect()

    print_memory_usage("After cleaning intermediate data")

    gdf_official_library, issues = validate_and_fix_data_integrity(gdf_official_library)
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    
    micro_anatomy(gdf_official_library)
    
    if gdf_official_library.crs is None:
        gdf_official_library = gdf_official_library.set_crs("EPSG:2326")
    
    gdf_official_library = fix_missing_distance_data(gdf_official_library, gdf_bdbiar)

    deep_validation(gdf_official_library)

    investigate_unmatched_bd(gdf_official_library, gdf_bdbiar)

    reverse_traceability(gdf_official_library, gdf_bdbiar)



    if not gdf_official_library.empty:
        gdf_official_library.to_file(OFFICIAL_LIB_BASE_PATH, driver='GeoJSON')
    else:
        pass

    if not gdf_official_library.empty:
        gdf_official_library, issues = validate_and_fix_data_integrity(gdf_official_library)
        
        if gdf_official_library.crs is None:
            gdf_official_library = gdf_official_library.set_crs("EPSG:2326")
        
        try:
            bounds = gdf_official_library.total_bounds
        except Exception as e:
            backup_path = OFFICIAL_LIBRARY_PATH.replace('.geojson', '_backup.geojson')
            gdf_official_library.to_file(backup_path, driver='GeoJSON')
        
        try:
            gdf_official_library.to_file(OFFICIAL_LIBRARY_PATH, driver='GeoJSON')
        except Exception as e:
            shp_path = OFFICIAL_LIBRARY_PATH.replace('.geojson', '.shp')
            gdf_official_library.to_file(shp_path)
    else:
        pass

    if not gdf_osm_buildings.empty or not gdf_osm_pois.empty:
        gdf_osm_all = pd.concat([gdf_osm_buildings, gdf_osm_pois], ignore_index=True)
        if 'id' in gdf_osm_all.columns:
            gdf_osm_all.rename(columns={'id': 'osmid'}, inplace=True)
        
        gdf_osm_all.to_file(OSM_ALL_PATH, driver='GeoJSON')
    else:
        pass


    from config import OVERTURE_PARQUET_FILE_PATH, OVERTURE_CLEAN_PATH
    
    gdf_overture = load_overture_places(OVERTURE_PARQUET_FILE_PATH)
    
    if not gdf_overture.empty:
        gdf_overture.to_file(OVERTURE_CLEAN_PATH, driver="GeoJSON")
    else:
        pass



    if not gdf_bdbiar.empty:
        gdf_bdbiar.to_file(BDBIAR_CACHE_PATH, driver='GeoJSON')
    else:
        pass

    if not os.path.exists(AGGREGATED_GDF_PATH):
        empty_gdf = gpd.GeoDataFrame(columns=['BUILDINGSTRUCTUREID', 'geometry'], crs="EPSG:2326")
        empty_gdf.to_file(AGGREGATED_GDF_PATH, driver='GeoJSON')
    else:
        pass
    
    if os.path.exists(OFFICIAL_LIB_BASE_PATH):
        pass
    else:
        pass
    
    
    
    
    final_memory = print_memory_usage("Final")
    if final_memory:
        pass
    
    return True

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        
        if '\r' not in message:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def isatty(self):
        return True

if __name__ == "__main__":
    log_dir = os.path.join(BASE_DIR, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "1.data_preprocessing.txt")
    
    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout
    
    
    try:
        success = main()
        if success:
            pass
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
