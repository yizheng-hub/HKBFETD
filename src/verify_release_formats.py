"""Read back every released attribute and geometry in all three formats."""
import json
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from config import OUTPUT_DIR

KEY = 'Building_ID_建筑ID'


def main():
    out = Path(OUTPUT_DIR)
    reference = gpd.read_parquet(out/'HKBFETD_v2.parquet').set_index(KEY)
    assert reference.index.is_unique
    differences = {}
    for extension in ['csv','geojson']:
        if extension == 'csv':
            frame = pd.read_csv(out/'HKBFETD_v2.csv',dtype={KEY:'string'},
                                keep_default_na=False,na_values=[''],low_memory=False).set_index(KEY)
        else:
            frame = gpd.read_file(out/'HKBFETD_v2.geojson').set_index(KEY)
            assert frame.crs.to_epsg() == 4326, 'Public GeoJSON must use RFC 7946 longitude/latitude'
        assert frame.index.is_unique and set(frame.index) == set(reference.index)
        frame = frame.reindex(reference.index)
        columns = reference.columns.drop('geometry')
        assert set(frame.columns)-{'geometry'} == set(columns)
        for column in columns:
            left,right = reference[column],frame[column]
            if pd.api.types.is_numeric_dtype(left):
                numeric = pd.to_numeric(right,errors='coerce')
                equal = np.isclose(left.to_numpy(dtype=float,na_value=np.nan),
                    numeric.to_numpy(dtype=float,na_value=np.nan),
                    equal_nan=True,rtol=1e-9,atol=1e-8)
                equal &= ~(right.notna() & numeric.isna()).to_numpy()
            else:
                equal = left.astype('string').fillna('').eq(right.astype('string').fillna('')).to_numpy()
            if not equal.all():
                differences[extension+':'+column] = int((~equal).sum())
        if extension == 'geojson':
            expected_geometry = reference.to_crs(4326).geometry.to_numpy()
            actual_geometry = frame.geometry.to_numpy()
            distances = shapely.hausdorff_distance(expected_geometry, actual_geometry)
            equal = distances <= 1e-9
            equal &= shapely.get_num_coordinates(expected_geometry) == shapely.get_num_coordinates(actual_geometry)
            equal &= shapely.get_type_id(expected_geometry) == shapely.get_type_id(actual_geometry)
            equal &= shapely.is_valid(actual_geometry)
            if not equal.all():
                differences['geojson:geometry'] = int((~equal).sum())
    result = {'records':len(reference),'formats_checked':3,'attributes_checked':len(columns),
              'id_set_and_uniqueness_checked':True,
              'parquet_geometry_crs':'EPSG:'+str(reference.crs.to_epsg()),
              'geojson_geometry_crs':'EPSG:4326', 'geojson_standard':'RFC 7946',
              'geometry_coordinate_tolerance_degrees':1e-9,
              'maximum_geometry_hausdorff_distance_degrees':float(distances.max()),
              'numeric_relative_tolerance':1e-9,'differences':differences}
    (out/'release_format_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    assert not differences, 'Release read-back differences: '+str(differences)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
