# osm_aggregation.py

"""Module documentation."""

# -*- coding: utf-8 -*-
import os
import sys
import warnings
import json
import re
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from pyproj import Transformer
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
import seaborn as sns
from shapely.geometry import Polygon, MultiPolygon, Point, box
from shapely.ops import unary_union

warnings.filterwarnings('ignore')

from config import (
    INTERMEDIATE_DIR, OUTPUT_DIR, OFFICIAL_LIBRARY_PATH, OSM_ALL_PATH,
    AGGREGATED_GDF_PATH, KEYWORDS_FILE, OSM_CLEAN_PATH,
    OVERLAP_RATIO_THRESHOLD, NUM_MULTIMATCH_SAMPLES, NUM_ISLAND_SAMPLES,
    RUN_AGGREGATION_EVALUATION, OVERTURE_CLEAN_PATH,
    ISLANDS_MAP_PATH, MULTI_OSM_VIS_PATH, MULTI_OSM_FACETED_PATH, VISUALIZATION_DPI
)

try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None
    print("[INFO] Status message emitted.")

from utils import (to_simplified_chinese)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
tqdm.pandas()

HK80_CRS = "EPSG:2326"
WGS84_CRS = "EPSG:4326"

def format_osmid(osmid_value):
    """Function documentation."""
    try:
        num_val = float(osmid_value)
        return f"{num_val:.0f}"
    except (ValueError, TypeError):
        return str(osmid_value)

def visualize_multi_osm_aggregation(gdf_aggregated, gdf_official_library, gdf_osm_clean, 
                                    multi_osm_buildings, num_samples=3):
    """Function documentation."""
    if multi_osm_buildings.empty or num_samples <= 0:
        return
    
    print("[INFO] Status message emitted.")
    
    matplotlib.use('Agg')
    
    try:
        colors = matplotlib.colormaps['tab20']
    except AttributeError:
        colors = plt.cm.get_cmap('tab20', 20)
    
    sample_ids = multi_osm_buildings.sample(min(num_samples, len(multi_osm_buildings)), random_state=42).index
    
    for bsid in sample_ids:
        official_geom = gdf_official_library[gdf_official_library['BUILDINGSTRUCTUREID'] == bsid].geometry.iloc[0]
        
        osm_records = gdf_aggregated[gdf_aggregated['BUILDINGSTRUCTUREID'] == bsid].dropna(subset=['osmid'])
        osm_ids = osm_records['osmid'].unique()
        
        osm_geoms_to_plot = gdf_osm_clean[
            gdf_osm_clean['osmid'].isin(osm_ids) & 
            gdf_osm_clean.geometry.notna() & 
            (~gdf_osm_clean.geometry.is_empty)
        ].copy()
        
        num_osm = len(osm_geoms_to_plot)
        
        print(f"\n" + "="*50)
        print("[INFO] Status message emitted.")
        
        if num_osm == 0:
            print("[INFO] Status message emitted.")
            continue
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        gpd.GeoSeries([official_geom], crs=gdf_official_library.crs).plot(
            ax=ax, color='gray', alpha=0.3, label='官方建筑'
        )
        
        for i, (_, r) in enumerate(osm_geoms_to_plot.iterrows()):
            color = colors(i % 20)
            label_text = f"OSM {i+1}: {format_osmid(r['osmid'])}"
            gpd.GeoSeries([r.geometry], crs=gdf_osm_clean.crs).plot(
                ax=ax, color=color, linewidth=2.5, 
                label=label_text
            )
        
        title = f"叠加视图: BSID {bsid} ({num_osm}个OSM)"
        if num_osm == 1:
            osm_class = osm_geoms_to_plot.iloc[0].get('initial_main_class', 'N/A')
            title += f"\nOSM Class: {osm_class}"
        
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_axis_off()
        
        output_path = MULTI_OSM_VIS_PATH.format(bsid)
        plt.tight_layout()
        plt.savefig(output_path, dpi=VISUALIZATION_DPI, bbox_inches='tight')
        plt.close()
        
        print("[INFO] Status message emitted.")
        
        if num_osm > 1:
            cols = min(num_osm, 4)
            rows = (num_osm + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
            axes = axes.flatten()
            
            for i, (_, r) in enumerate(osm_geoms_to_plot.iterrows()):
                ax = axes[i]
                color = colors(i % 20)
                
                gpd.GeoSeries([official_geom], crs=gdf_official_library.crs).plot(
                    ax=ax, color='#e0e0e0', alpha=0.5
                )
                
                gpd.GeoSeries([r.geometry], crs=gdf_osm_clean.crs).plot(
                    ax=ax, color=color, linewidth=3
                )
                
                osm_id = r['osmid']
                osm_main_class = r.get('initial_main_class', 'N/A')
                osm_sub_class = r.get('initial_sub_class', 'N/A')
                
                title = f"OSM {i+1}: {format_osmid(osm_id)}\n主类: {osm_main_class}\n子类: {osm_sub_class}"
                
                ax.set_title(title, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_axis_off()
            
            for i in range(num_osm, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            faceted_path = MULTI_OSM_FACETED_PATH.format(bsid)
            plt.savefig(faceted_path, dpi=VISUALIZATION_DPI, bbox_inches='tight')
            plt.close()
            
            print("[INFO] Status message emitted.")
        
        print("="*50)

def visualize_islands_on_map(island_gdf, num_samples=100):
    """Function documentation."""
    if island_gdf.empty:
        print("[INFO] Status message emitted.")
        return
    
    print("[INFO] Status message emitted.")
    
    samples = island_gdf.sample(min(num_samples, len(island_gdf)), random_state=42)
    
    transformer = Transformer.from_crs(HK80_CRS, WGS84_CRS, always_xy=True)
    
    try:
        centroid = samples.geometry.representative_point().unary_union.centroid
        center_lon, center_lat = transformer.transform(centroid.x, centroid.y)
    except Exception as e:
        print("[INFO] Status message emitted.")
        center_lat, center_lon = 22.35, 114.1
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, 
                   tiles="CartoDB positron")
    
    folium.TileLayer(
        'https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=True
    ).add_to(m)
    
    for _, row in tqdm(samples.iterrows(), total=len(samples), desc="添加数据孤岛样本"):
        if row.geometry is None or row.geometry.is_empty:
            continue
        
        try:
            if row.geometry.geom_type == 'Polygon':
                coords = []
                for x, y in row.geometry.exterior.coords:
                    lon, lat = transformer.transform(x, y)
                    coords.append([lat, lon])
                
                popup_html = f"""
                <h4>数据孤岛样本</h4><hr>
                <b>BSID:</b> {row['BUILDINGSTRUCTUREID']}<br>
                <b>面积:</b> {row.geometry.area:.2f} 平方米<br>
                <b>类型:</b> {row.get('BUILDINGSTRUCTURETYPE', 'N/A')}
                """
                
                folium.Polygon(
                    locations=coords,
                    popup=folium.Popup(popup_html, max_width=300),
                    color='yellow',
                    weight=2,
                    fill=True,
                    fill_color='yellow',
                    fill_opacity=0.7,
                    tooltip=f"BSID: {row['BUILDINGSTRUCTUREID']}"
                ).add_to(m)
                
            elif row.geometry.geom_type == 'Point':
                lon, lat = transformer.transform(row.geometry.x, row.geometry.y)
                
                popup_html = f"""
                <h4>数据孤岛样本 (点)</h4><hr>
                <b>BSID:</b> {row['BUILDINGSTRUCTUREID']}<br>
                <b>类型:</b> {row.get('BUILDINGSTRUCTURETYPE', 'N/A')}
                """
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    popup=folium.Popup(popup_html, max_width=300),
                    color='yellow',
                    fill=True,
                    fill_color='yellow',
                    fill_opacity=0.8,
                    tooltip=f"BSID: {row['BUILDINGSTRUCTUREID']}"
                ).add_to(m)
                
        except Exception as e:
            print("[INFO] Status message emitted.")
            continue
    
    m.save(ISLANDS_MAP_PATH)
    print("[INFO] Status message emitted.")

def aggregation_evaluation(gdf_aggregated, gdf_official_library, gdf_osm_clean):
    """Function documentation."""
    if not RUN_AGGREGATION_EVALUATION:
        return
    
    print("[INFO] Status message emitted.")
    
    print("[INFO] Status message emitted.")
    
    total_buildings = len(gdf_official_library)
    print("[INFO] Status message emitted.")
    
    print("[INFO] Status message emitted.")
    
    if 'osmid' in gdf_aggregated.columns:
        matched_mask = gdf_aggregated['osmid'].notna()
        if matched_mask.any():
            matched_building_ids = gdf_aggregated[matched_mask]['BUILDINGSTRUCTUREID'].unique()
            matched_building_count = len(matched_building_ids)
        else:
            matched_building_ids = []
            matched_building_count = 0
    else:
        matched_building_count = 0
    
    unmatched_building_count = total_buildings - matched_building_count
    
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    
    print("[INFO] Status message emitted.")
    
    if matched_building_count > 0 and 'osmid' in gdf_aggregated.columns:
        osm_counts_per_building = gdf_aggregated[gdf_aggregated['osmid'].notna()].groupby('BUILDINGSTRUCTUREID')['osmid'].nunique()
        
        multi_osm_buildings = osm_counts_per_building[osm_counts_per_building > 1]
        num_multi_osm = len(multi_osm_buildings)
        
        print("[INFO] Status message emitted.")
        print("[INFO] Status message emitted.")
        print("[INFO] Status message emitted.")
        
        if num_multi_osm > 0:
            print("[INFO] Status message emitted.")
            print(multi_osm_buildings.describe().to_string())
            
            print("[INFO] Status message emitted.")
            top_5 = multi_osm_buildings.sort_values(ascending=False).head(5)
            for idx, count in top_5.items():
                building_info = gdf_official_library[
                    gdf_official_library['BUILDINGSTRUCTUREID'] == idx
                ]
                if not building_info.empty:
                    name = building_info.iloc[0].get('BUILDINGNAMETC', 'N/A')
                    area = building_info.iloc[0].geometry.area
                    print("[INFO] Status message emitted.")
                else:
                    print("[INFO] Status message emitted.")
    else:
        print("[INFO] Status message emitted.")
        multi_osm_buildings = pd.Series(dtype=int)
    
    print("[INFO] Status message emitted.")
    
    island_criteria = pd.Series(False, index=gdf_aggregated.index)
    
    has_osmid = 'osmid' in gdf_aggregated.columns
    has_bdbiar = 'BDBIAR_OBJECTID' in gdf_aggregated.columns
    has_name = 'BUILDINGNAMETC' in gdf_aggregated.columns
    has_desc = 'ALL_INFO_DESC' in gdf_aggregated.columns
    
    if has_osmid:
        island_criteria = island_criteria | gdf_aggregated['osmid'].isna()
    else:
        island_criteria = island_criteria | True
    
    if has_bdbiar:
        island_criteria = island_criteria & gdf_aggregated['BDBIAR_OBJECTID'].isna()
    else:
        island_criteria = island_criteria & True
    
    if has_name:
        island_criteria = island_criteria & gdf_aggregated['BUILDINGNAMETC'].isna()
    else:
        island_criteria = island_criteria & True
    
    if has_desc:
        island_criteria = island_criteria & gdf_aggregated['ALL_INFO_DESC'].isna()
    else:
        island_criteria = island_criteria & True
    
    information_islands = gdf_aggregated[island_criteria].copy()
    num_islands = len(information_islands)
    
    if total_buildings > 0:
        print("[INFO] Status message emitted.")
        print("[INFO] Status message emitted.")
    
    print("[INFO] Status message emitted.")
    
    if 'multi_osm_buildings' in locals() and not multi_osm_buildings.empty:
        visualize_multi_osm_aggregation(
            gdf_aggregated, 
            gdf_official_library,
            gdf_osm_clean,
            multi_osm_buildings,
            num_samples=NUM_MULTIMATCH_SAMPLES
        )
    else:
        print("[INFO] Status message emitted.")
    
    if not information_islands.empty:
        visualize_islands_on_map(
            information_islands, 
            num_samples=min(NUM_ISLAND_SAMPLES, len(information_islands))
        )
    else:
        print("[INFO] Status message emitted.")
    
    print("[INFO] Status message emitted.")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        labels = ['匹配OSM', '未匹配OSM']
        sizes = [matched_building_count, unmatched_building_count]
        colors = ['#66c2a5', '#fc8d62']
        
        axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('OSM匹配情况分布')
        
        if 'multi_osm_buildings' in locals() and not multi_osm_buildings.empty:
            display_data = multi_osm_buildings[multi_osm_buildings <= 20]
            if not display_data.empty:
                axes[1].hist(display_data, bins=20, color='#8da0cb', edgecolor='black')
                axes[1].set_xlabel('匹配的OSM要素数量')
                axes[1].set_ylabel('建筑数量')
                axes[1].set_title('一对多匹配分布 (≤20个OSM)')
            else:
                axes[1].text(0.5, 0.5, '无一对多匹配数据', ha='center', va='center')
        else:
            axes[1].text(0.5, 0.5, '无一对多匹配数据', ha='center', va='center')
        
        plt.tight_layout()
        stats_path = os.path.join(OUTPUT_DIR, "aggregation_statistics.png")
        plt.savefig(stats_path, dpi=VISUALIZATION_DPI)
        plt.close()
        
        print("[INFO] Status message emitted.")
        
    except Exception as e:
        print("[INFO] Status message emitted.")
    
    print("[INFO] Status message emitted.")

def debug_geometry_info(gdf, label=""):
    """Function documentation."""
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    if not gdf.empty:
        print(f"  - CRS: {gdf.crs}")
        print("[INFO] Status message emitted.")
        try:
            bbox = gdf.total_bounds
            print("[INFO] Status message emitted.")
            print(f"    x: {bbox[0]:.2f} ~ {bbox[2]:.2f}")
            print(f"    y: {bbox[1]:.2f} ~ {bbox[3]:.2f}")
        except:
            print("[INFO] Status message emitted.")
        print("[INFO] Status message emitted.")
        print("[INFO] Status message emitted.")

def simple_geometry_validation(gdf, label="数据"):
    """Function documentation."""
    print("[INFO] Status message emitted.")
    
    if gdf.empty:
        print("[INFO] Status message emitted.")
        return gdf
    
    original_count = len(gdf)
    
    invalid_mask = ~gdf.geometry.is_valid
    invalid_count = invalid_mask.sum()
    
    if invalid_count > 0:
        print("[INFO] Status message emitted.")
        
        def simple_repair(geom):
            """Function documentation."""
            try:
                if geom is None or geom.is_empty:
                    return geom
                try:
                    fixed = geom.buffer(0)
                    if fixed.is_valid and not fixed.is_empty:
                        return fixed
                except:
                    pass
                
                try:
                    simplified = geom.simplify(0.001, preserve_topology=True)
                    if simplified.is_valid and not simplified.is_empty:
                        return simplified
                except:
                    pass
                
                return geom
            except:
                return geom
        
        gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask, 'geometry'].apply(simple_repair)
    
    still_invalid = ~gdf.geometry.is_valid
    still_invalid_count = still_invalid.sum()
    
    if still_invalid_count > 0:
        print("[INFO] Status message emitted.")
    
    print("[INFO] Status message emitted.")
    
    return gdf

def fuse_poi_datasets(gdf_osm_pois, gdf_overture):
    """Function documentation."""
    print("[INFO] Status message emitted.")
    if gdf_overture is None or gdf_overture.empty:
        print("[INFO] Status message emitted.")
        return gdf_osm_pois
        
    print("[INFO] Status message emitted.")
    
    ovt_buffer = gdf_overture.copy()
    ovt_buffer['geometry'] = ovt_buffer.geometry.buffer(10)
    
    joined = gpd.sjoin(gdf_osm_pois, ovt_buffer, how='inner', predicate='within')
    
    duplicates_osm_idx = set()
    if fuzz is not None:
        print("[INFO] Status message emitted.")
        for idx, row in joined.iterrows():
            name_osm = str(row.get('name_left', '')).lower()
            name_ovt = str(row.get('name_right', '')).lower()
            
            if name_osm and name_ovt:
                if fuzz.token_set_ratio(name_osm, name_ovt) > 80:
                    duplicates_osm_idx.add(idx)
            elif not name_osm and not name_ovt:
                if row.get('amenity_left') == row.get('amenity_right') or row.get('shop_left') == row.get('shop_right'):
                    duplicates_osm_idx.add(idx)
    else:
        print("[INFO] Status message emitted.")
        for idx, row in joined.iterrows():
            if row.get('amenity_left') == row.get('amenity_right') or row.get('shop_left') == row.get('shop_right'):
                duplicates_osm_idx.add(idx)
    
    valid_drop_indices = [i for i in duplicates_osm_idx if i in gdf_osm_pois.index]
    gdf_osm_unique = gdf_osm_pois.drop(index=valid_drop_indices)
    print("[INFO] Status message emitted.")
    
    if 'data_source' not in gdf_osm_unique.columns:
        gdf_osm_unique['data_source'] = 'osm'
    if 'data_source' not in gdf_overture.columns:
        gdf_overture['data_source'] = 'overture'
        
    cols_to_keep = ['osmid', 'name', 'amenity', 'shop', 'data_source', 'geometry']
    common_cols_osm = [c for c in cols_to_keep if c in gdf_osm_unique.columns]
    common_cols_ovt = [c for c in cols_to_keep if c in gdf_overture.columns]
    
    gdf_fused = pd.concat([gdf_overture[common_cols_ovt], gdf_osm_unique[common_cols_osm]], ignore_index=True)
    gdf_fused = gpd.GeoDataFrame(gdf_fused, geometry='geometry', crs=gdf_osm_pois.crs)
    print("[INFO] Status message emitted.")
    
    return gdf_fused



def load_input_data():
    """Function documentation."""
    print("[INFO] Status message emitted.")
    
    try:
        print("[INFO] Status message emitted.")
        if os.path.exists(OFFICIAL_LIBRARY_PATH):
            gdf_official_library = gpd.read_file(OFFICIAL_LIBRARY_PATH)
            
            print("[INFO] Status message emitted.")
            print("[INFO] Status message emitted.")
            print("[INFO] Status message emitted.")
            
            if gdf_official_library.empty:
                print("[INFO] Status message emitted.")
                return None
            
            if gdf_official_library.crs is None:
                print("[INFO] Status message emitted.")
                gdf_official_library = gdf_official_library.set_crs(HK80_CRS)
            elif str(gdf_official_library.crs) != HK80_CRS:
                print("[INFO] Status message emitted.")
                gdf_official_library = gdf_official_library.to_crs(HK80_CRS)
            
            print("[INFO] Status message emitted.")
            gdf_official_library = simple_geometry_validation(gdf_official_library, "官方建筑库")
                
        else:
            print("[INFO] Status message emitted.")
            print("[INFO] Status message emitted.")
            return None
        
        print("[INFO] Status message emitted.")
        if os.path.exists(OSM_ALL_PATH):
            gdf_osm_all = gpd.read_file(OSM_ALL_PATH)
            
            print("[INFO] Status message emitted.")
            print("[INFO] Status message emitted.")
            print(f"     - CRS: {gdf_osm_all.crs}")
            
            if not gdf_osm_all.empty:
                if gdf_osm_all.crs is None:
                    print("[INFO] Status message emitted.")
                    gdf_osm_all = gdf_osm_all.set_crs(WGS84_CRS)
                
                print("[INFO] Status message emitted.")
                try:
                    gdf_osm_all = gdf_osm_all.to_crs(HK80_CRS)
                except Exception as e:
                    print("[INFO] Status message emitted.")
                    print("[INFO] Status message emitted.")
                    gdf_osm_all = gdf_osm_all.set_crs(HK80_CRS)
                
                print("[INFO] Status message emitted.")
                gdf_osm_buildings = gdf_osm_all[
                    gdf_osm_all.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
                ].copy()
                gdf_osm_pois = gdf_osm_all[
                    gdf_osm_all.geometry.geom_type.isin(['Point'])
                ].copy()
                
                print("[INFO] Status message emitted.")

                print("[INFO] Status message emitted.")
                try:
                    if os.path.exists(OVERTURE_CLEAN_PATH):
                        gdf_overture = gpd.read_file(OVERTURE_CLEAN_PATH)
                        gdf_osm_pois = fuse_poi_datasets(gdf_osm_pois, gdf_overture)
                    else:
                        print("[INFO] Status message emitted.")
                except Exception as e:
                    print("[INFO] Status message emitted.")
                # =================================================================

            else:
                print("[INFO] Status message emitted.")
                gdf_osm_buildings = gpd.GeoDataFrame()
                gdf_osm_pois = gpd.GeoDataFrame()
        else:
            print("[INFO] Status message emitted.")
            print("[INFO] Status message emitted.")
            return None
        
        if not gdf_official_library.empty and not gdf_osm_buildings.empty:
            print("[INFO] Status message emitted.")
            try:
                bounds1 = gdf_official_library.total_bounds
                bounds2 = gdf_osm_buildings.total_bounds
                
                print("[INFO] Status message emitted.")
                print("[INFO] Status message emitted.")
                
                overlap_x = not (bounds1[2] < bounds2[0] or bounds1[0] > bounds2[2])
                overlap_y = not (bounds1[3] < bounds2[1] or bounds1[1] > bounds2[3])
                
                if overlap_x and overlap_y:
                    print("[INFO] Status message emitted.")
                else:
                    print("[INFO] Status message emitted.")
            except Exception as e:
                print("[INFO] Status message emitted.")
        
        return {
            'gdf_official_library': gdf_official_library,
            'gdf_osm_buildings': gdf_osm_buildings,
            'gdf_osm_pois': gdf_osm_pois,
            'gdf_osm_all': gdf_osm_all
        }
        
    except Exception as e:
        print("[INFO] Status message emitted.")
        import traceback
        traceback.print_exc()
        return None

def preprocess_osm(gdf_osm_buildings, gdf_osm_pois, gdf_official_library):
    """Function documentation."""
    print("[INFO] Status message emitted.")
    
    print("[INFO] Status message emitted.")
    try:
        from config import KEYWORDS_FILE
        with open(KEYWORDS_FILE, 'r', encoding='utf-8') as f:
            KEYWORDS_CONFIG = json.load(f)
    except Exception as e:
        raise IOError(f"错误: 无法加载关键词文件 '{KEYWORDS_FILE}': {e}")
    
    KEYWORD_MAP = {}
    
    for main, subs in KEYWORDS_CONFIG.items():
        if not isinstance(subs, dict):
            continue
            
        for sub, kws in subs.items():
            for kw in kws:
                if kw_sl := to_simplified_chinese(kw).lower().strip():
                    KEYWORD_MAP[kw_sl] = (main, sub)
    
    all_kws_list = sorted([re.escape(k) for k in KEYWORD_MAP.keys() if len(k) > 1], key=len, reverse=True)
    KEYWORD_REGEX = re.compile('|'.join(all_kws_list)) if all_kws_list else None
    print("[INFO] Status message emitted.")
    
    def classify_text_for_osm(text):
        if not text or pd.isna(text) or not KEYWORD_REGEX:
            return [], []
        matches = KEYWORD_REGEX.findall(to_simplified_chinese(text).lower().strip())
        if matches:
            return list(set(matches)), list(set(KEYWORD_MAP[kw] for kw in matches))
        return [], []
    
    def classify_osm_feature_for_join(row):
        text_parts = []
        for col in ['shop', 'amenity', 'building', 'name']:
            if col in row and pd.notna(row[col]):
                text_parts.append(str(row[col]))
        text = ' '.join(text_parts)
        _, classifications = classify_text_for_osm(text)
        if classifications:
            return classifications[0]
        return (None, None)
    
    print("[INFO] Status message emitted.")
    
    gdf_osm_all = pd.concat([gdf_osm_buildings, gdf_osm_pois], ignore_index=True)
    
    if gdf_osm_all.empty:
        print("[INFO] Status message emitted.")
        gdf_osm_clean = gpd.GeoDataFrame(
            columns=['osmid', 'name', 'building', 'amenity', 'shop', 
                     'geometry', 'osm_height', 'osm_area', 
                     'initial_main_class', 'initial_sub_class'],
            crs=HK80_CRS
        )
        gdf_osm_clean.to_file(OSM_CLEAN_PATH, driver='GeoJSON')
        print("[INFO] Status message emitted.")
        return gdf_osm_clean
    
    if 'id' in gdf_osm_all.columns:
        gdf_osm_all.rename(columns={'id': 'osmid'}, inplace=True)
    elif 'osmid' not in gdf_osm_all.columns:
        gdf_osm_all['osmid'] = range(len(gdf_osm_all))
    
    gdf_osm_all = simple_geometry_validation(gdf_osm_all, "OSM预处理")
    
    print("[INFO] Status message emitted.")
    if gdf_osm_all.crs is None:
        gdf_osm_all = gdf_osm_all.set_crs(HK80_CRS)
    elif str(gdf_osm_all.crs) != HK80_CRS:
        gdf_osm_all = gdf_osm_all.to_crs(HK80_CRS)
    
    osm_base_tags = ['osmid', 'name', 'building', 'amenity', 'shop', 'geometry']
    available_columns = [c for c in osm_base_tags if c in gdf_osm_all.columns]
    gdf_osm_clean = gdf_osm_all[available_columns].copy()
    
    if 'building:height' in gdf_osm_all.columns:
        gdf_osm_clean['osm_height'] = pd.to_numeric(gdf_osm_all['building:height'], errors='coerce')
        print("[INFO] Status message emitted.")
    else:
        gdf_osm_clean['osm_height'] = np.nan
        print("[INFO] Status message emitted.")
    
    gdf_osm_clean['osm_area'] = gdf_osm_clean.geometry.area
    
    print("[INFO] Status message emitted.")
    if len(gdf_osm_clean) > 0:
        tqdm.pandas(desc="OSM初步分类")
        osm_initial_classes = gdf_osm_clean.progress_apply(classify_osm_feature_for_join, axis=1)
        gdf_osm_clean['initial_main_class'] = [c[0] for c in osm_initial_classes]
        gdf_osm_clean['initial_sub_class'] = [c[1] for c in osm_initial_classes]
        print("[INFO] Status message emitted.")
    else:
        gdf_osm_clean['initial_main_class'] = None
        gdf_osm_clean['initial_sub_class'] = None
        print("[INFO] Status message emitted.")
    
    total_osm_features = len(gdf_osm_clean)
    if total_osm_features > 0:
        classified_osm = gdf_osm_clean.dropna(subset=['initial_main_class'])
        num_classified = len(classified_osm)
        num_unclassified = total_osm_features - num_classified
        
        print("[INFO] Status message emitted.")
        print("[INFO] Status message emitted.")
        print("[INFO] Status message emitted.")
        print("[INFO] Status message emitted.")
        
        if not classified_osm.empty:
            print("[INFO] Status message emitted.")
            class_counts = classified_osm['initial_main_class'].value_counts().head(10)
            print(class_counts.to_markdown())
    else:
        print("[INFO] Status message emitted.")
        print("[INFO] Status message emitted.")
    
    gdf_osm_clean.to_file(OSM_CLEAN_PATH, driver='GeoJSON')
    print("[INFO] Status message emitted.")
    
    debug_geometry_info(gdf_osm_clean, "清理后的OSM数据")
    
    return gdf_osm_clean

def spatial_aggregation(gdf_official_library, gdf_osm_clean):
    """Function documentation."""
    print("[INFO] Status message emitted.")
    
    print("[INFO] Status message emitted.")

    print("[INFO] Status message emitted.")
    
    if len(gdf_official_library) == 0:
        print("[INFO] Status message emitted.")
        return gdf_official_library
    
    if len(gdf_osm_clean) == 0:
        print("[INFO] Status message emitted.")
        gdf_aggregated = gdf_official_library.copy()
        osm_columns = ['osmid', 'name', 'building', 'amenity', 'shop', 
                      'initial_main_class', 'initial_sub_class', 'osm_height', 
                      'osm_area', 'overlap_ratio', 'index_right', 'osm_weight_score']
        for col in osm_columns:
            gdf_aggregated[col] = np.nan
        return gdf_aggregated
    
    debug_geometry_info(gdf_official_library, "空间聚合前的官方建筑库")
    debug_geometry_info(gdf_osm_clean, "空间聚合前的OSM数据")
    
    print("[INFO] Status message emitted.")
    
    if str(gdf_official_library.crs) != str(gdf_osm_clean.crs):
        print("[INFO] Status message emitted.")
        print("[INFO] Status message emitted.")
        gdf_osm_clean = gdf_osm_clean.to_crs(gdf_official_library.crs)
    
    try:
        potential_matches = gpd.sjoin(
            gdf_official_library, 
            gdf_osm_clean, 
            how="inner", 
            predicate='intersects'
        )
        print("[INFO] Status message emitted.")
    except Exception as e:
        print("[INFO] Status message emitted.")
        print("[INFO] Status message emitted.")
        
        batch_size = 50000
        potential_matches_list = []
        
        for i in range(0, len(gdf_official_library), batch_size):
            batch = gdf_official_library.iloc[i:i+batch_size]
            try:
                batch_matches = gpd.sjoin(batch, gdf_osm_clean, how="inner", predicate='intersects')
                potential_matches_list.append(batch_matches)
                print("[INFO] Status message emitted.")
            except Exception as e2:
                print("[INFO] Status message emitted.")
        
        if potential_matches_list:
            potential_matches = pd.concat(potential_matches_list, ignore_index=True)
            print("[INFO] Status message emitted.")
        else:
            print("[INFO] Status message emitted.")
            gdf_aggregated = gdf_official_library.copy()
            osm_columns = ['osmid', 'name', 'building', 'amenity', 'shop', 
                          'initial_main_class', 'initial_sub_class', 'osm_height', 
                          'osm_area', 'overlap_ratio', 'index_right', 'osm_weight_score']
            for col in osm_columns:
                gdf_aggregated[col] = np.nan
            return gdf_aggregated
    
    print("[INFO] Status message emitted.")
    
    if len(potential_matches) > 0:
        print("[INFO] Status message emitted.")
        osm_geometry_dict = {idx: row.geometry for idx, row in gdf_osm_clean.iterrows()}
        
        def calculate_overlap_ratio_safe(row):
            """Function documentation."""
            try:
                if 'index_right' in row and pd.notna(row['index_right']):
                    idx = int(row['index_right'])
                    if idx in osm_geometry_dict:
                        osm_geom = osm_geometry_dict[idx]
                        official_geom = row.geometry
                        
                        if osm_geom is not None and official_geom is not None:
                            if osm_geom.is_valid and official_geom.is_valid:
                                
                                if osm_geom.geom_type == 'Point':
                                    return 1.0
                                    
                                intersection_area = osm_geom.intersection(official_geom).area
                                osm_area = osm_geom.area
                                if osm_area > 1e-6:
                                    return intersection_area / osm_area
            except Exception as e:
                pass
            return 0.0
        
        tqdm.pandas(desc="计算重叠比例")
        potential_matches['overlap_ratio'] = potential_matches.progress_apply(
            calculate_overlap_ratio_safe, axis=1
        )
        
        strong_matches = potential_matches[potential_matches['overlap_ratio'] >= OVERLAP_RATIO_THRESHOLD].copy()
        print("[INFO] Status message emitted.")
    else:
        strong_matches = potential_matches.copy()
        print("[INFO] Status message emitted.")
    
    print("[INFO] Status message emitted.")
    
    if len(strong_matches) > 0:
        print("[INFO] Status message emitted.")
        
        # =========================================================
        # =========================================================
        strong_matches['osm_raw_area'] = strong_matches['osm_area'].fillna(0)
        strong_matches['official_area'] = strong_matches.geometry.area
        
        POINT_INFLUENCE_AREA = 50.0 
        
        def calculate_osm_weight(row):
            if row['osm_raw_area'] > 1: # Polygon
                return min(row['overlap_ratio'], 1.0)
            else: # Point
                if row['official_area'] > 0:
                    return min(POINT_INFLUENCE_AREA / row['official_area'], 0.3) 
                return 0.1
        
        strong_matches['osm_weight_score'] = strong_matches.apply(calculate_osm_weight, axis=1)
        # =========================================================

        
        matched_building_ids = strong_matches['BUILDINGSTRUCTUREID'].unique()
        
        unmatched_buildings = gdf_official_library[
            ~gdf_official_library['BUILDINGSTRUCTUREID'].isin(matched_building_ids)
        ].copy()
        
        osm_columns = ['osmid', 'name', 'building', 'amenity', 'shop',
                       'initial_main_class', 'initial_sub_class', 'osm_height', 
                       'osm_area', 'overlap_ratio', 'index_right', 'osm_weight_score']
        for col in osm_columns:
            unmatched_buildings[col] = np.nan
        
        gdf_aggregated = pd.concat([strong_matches, unmatched_buildings], ignore_index=True)
        
        print("[INFO] Status message emitted.")
        print("[INFO] Status message emitted.")
        print("[INFO] Status message emitted.")
        
        building_osm_counts = strong_matches.groupby('BUILDINGSTRUCTUREID')['osmid'].nunique()
        multi_match_count = (building_osm_counts > 1).sum()
        print("[INFO] Status message emitted.")
    else:
        gdf_aggregated = gdf_official_library.copy()
        osm_columns = ['osmid', 'name', 'building', 'amenity', 'shop', 
                      'initial_main_class', 'initial_sub_class', 'osm_height', 
                      'osm_area', 'overlap_ratio', 'index_right', 'osm_weight_score']
        for col in osm_columns:
            gdf_aggregated[col] = np.nan
    
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    
    original_ids = set(gdf_official_library['BUILDINGSTRUCTUREID'].unique())
    aggregated_ids = set(gdf_aggregated['BUILDINGSTRUCTUREID'].unique())
    
    missing_ids = original_ids - aggregated_ids
    extra_ids = aggregated_ids - original_ids
    
    if len(missing_ids) == 0:
        print("[INFO] Status message emitted.")
    else:
        print("[INFO] Status message emitted.")
    
    if len(extra_ids) == 0:
        print("[INFO] Status message emitted.")
    else:
        print("[INFO] Status message emitted.")
    
    building_counts = gdf_aggregated['BUILDINGSTRUCTUREID'].value_counts()
    multi_match_buildings = building_counts[building_counts > 1]
    
    if len(multi_match_buildings) > 0:
        print("[INFO] Status message emitted.")
        print("[INFO] Status message emitted.")
        print("[INFO] Status message emitted.")
        
        print("[INFO] Status message emitted.")
        for count in range(2, 11):
            num_buildings = (building_counts == count).sum()
            if num_buildings > 0:
                print("[INFO] Status message emitted.")
        
        if building_counts.max() > 10:
            num_extreme = (building_counts > 10).sum()
            print("[INFO] Status message emitted.")
    else:
        print("[INFO] Status message emitted.")


    if 'osmid' in gdf_aggregated.columns:
        matched_count = gdf_aggregated['osmid'].notna().sum()
        print("[INFO] Status message emitted.")
    
    print("[INFO] Status message emitted.")
    try:
        gdf_aggregated.to_file(AGGREGATED_GDF_PATH, driver='GeoJSON')
        print("[INFO] Status message emitted.")
    except Exception as e:
        print("[INFO] Status message emitted.")
        print("[INFO] Status message emitted.")
        try:
            csv_path = AGGREGATED_GDF_PATH.replace('.geojson', '.csv')
            df_without_geom = gdf_aggregated.drop(columns=['geometry'])
            df_without_geom.to_csv(csv_path, index=False, encoding='utf-8')
            print("[INFO] Status message emitted.")
        except Exception as e2:
            print("[INFO] Status message emitted.")
    
    return gdf_aggregated

def main():
    """Function documentation."""
    print("="*60)
    print("[INFO] Status message emitted.")
    print("="*60)
    
    data_dict = load_input_data()
    if data_dict is None:
        print("[INFO] Status message emitted.")
        return False
    
    gdf_official_library = data_dict['gdf_official_library']
    gdf_osm_buildings = data_dict['gdf_osm_buildings']
    gdf_osm_pois = data_dict['gdf_osm_pois']
    
    gdf_osm_clean = preprocess_osm(
        gdf_osm_buildings, 
        gdf_osm_pois, 
        gdf_official_library
    )
    
    gdf_aggregated = spatial_aggregation(gdf_official_library, gdf_osm_clean)
    
    aggregation_evaluation(gdf_aggregated, gdf_official_library, gdf_osm_clean)
    
    print("\n" + "="*60)
    print("[INFO] Status message emitted.")
    print("="*60)
    
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(os.path.dirname(base_dir), "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "2.osm_aggregation.txt")
    
    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout
    print("[INFO] Status message emitted.")

    try:
        success = main()
        if success:
            print("[INFO] Status message emitted.")
        else:
            print("[INFO] Status message emitted.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("[INFO] Status message emitted.")
        sys.exit(0)
    except Exception as e:
        print("[INFO] Status message emitted.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
