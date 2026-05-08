from shapely.geometry import shape, Polygon
import geopandas as gpd
import numpy as np
from math import pi, sqrt
import glob
import os

def compactness_ratio(polygon):
    """Calculate the compactness ratio of a polygon."""
    if polygon.is_empty:
        return None  # Handle empty polygons

    area = polygon.area
    perimeter = polygon.length
    
    if perimeter == 0:
        return None  # Avoid division by zero when perimeter is zero

    return (4 * np.pi * area) / (perimeter ** 2)

def shape_index(polygon):
    """Calculate the shape index of a polygon: Perimeter / (2 * sqrt(pi * Area))."""
    if polygon.is_empty or polygon.area == 0:
        return None  # Handle empty polygons and avoid division by zero
    return polygon.length / (2 * np.sqrt(np.pi * polygon.area))

def interior_edge_ratio(polygon):
    """Calculate the interior edge ratio of a polygon: Perimeter / Area."""
    if polygon.is_empty or polygon.area == 0:
        return None  # Handle empty polygons and avoid division by zero
    return polygon.length / polygon.area

def calculate_fractal_dimension(polygon):
    """Calculate the fractal dimension of a polygon using the perimeter-area relationship."""
    perimeter = polygon.length
    area = polygon.area
    if area > 0 and perimeter > 0:
        return (2 * np.log(perimeter)) / np.log(area)
    else:
        return 0


path = '/data/Aldhani/cv_fields/preds/descartes_tiles/inference/mozambique/2023/post/'

# Specify the path to the GeoPackage file
files = glob.glob(path + '*_cleaned.gpkg')
print(len(files))

for file in files:
    output_file_path = file[:-5] + '_calc.gpkg'
    if not os.path.isfile(output_file_path):
        print(f"{output_file_path}.")
        gdf = gpd.read_file(file)
        if len(gdf) > 0:
            #print(f"{output_file_path}.")
            gdf['area'] = gdf.geometry.area
            gdf['perimeter'] = gdf.geometry.length

            # Calculate the interior edge ratio for each polygon
            gdf['interior_edge_ratio'] = gdf['geometry'].apply(interior_edge_ratio)

            # Calculate the shape index for each polygon
            gdf['shape_index'] = gdf['geometry'].apply(shape_index)

            # Calculate the compactness ratio for each polygon
            gdf['compactness_ratio'] = gdf['geometry'].apply(compactness_ratio)

            # Calculate the fractal dimension for each polygon
            gdf['fractal_dimension'] = gdf['geometry'].apply(calculate_fractal_dimension)

            # Display the results
            gdf.to_file(output_file_path, driver='GPKG')
