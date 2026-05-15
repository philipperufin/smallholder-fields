import pandas as pd
import numpy as np
import os
import glob
from osgeo import ogr, osr
from osgeo import gdal
from tqdm import tqdm
from shapely.ops import unary_union
from shapely.wkb import loads
from shapely.geometry import Polygon, MultiPolygon
import gc
import time

gdal.UseExceptions()

####################
# getting corresponding files for moving window

# function to find corresponding files for moving window based on ti and tj ids
def find_files(path, ti, tj):
    '''
    Function to find corresponding file paths for moving window.
    
    Input: 
        path = r string; path to folder where field tiles are located
        ti = int; corresponding ti for center tile in moving window
        tj = int; corresponding tj for center tile in moving window
        epsg = int; 32636 or 32637
        
    Output: 
        matching_files1 = list; file paths for 5x5 moving window
        matching_files2 = list; file paths for center 3x3 window
    '''

    # 5x5 moving window border tiles for specific ti and tj
    ti_tj_ls1 = [f"{ti-2}_{tj+2}",
                f"{ti-1}_{tj+2}",
                f"{ti}_{tj+2}",
                f"{ti+1}_{tj+2}",
                f"{ti+2}_{tj+2}",
                f"{ti-2}_{tj+1}",
                f"{ti+2}_{tj+1}",
                f"{ti-2}_{tj}",
                f"{ti+2}_{tj}",
                f"{ti-2}_{tj-1}",
                f"{ti+2}_{tj-1}",
                f"{ti-2}_{tj-2}",
                f"{ti-1}_{tj-2}",
                f"{ti}_{tj-2}",
                f"{ti+1}_{tj-2}",
                f"{ti+2}_{tj-2}"
                ]

    # 3x3 tiles mowing window border tiles for specific ti and tj
    ti_tj_ls2 = [f"{ti-1}_{tj+1}",
                f"{ti}_{tj+1}",
                f"{ti+1}_{tj+1}",
                f"{ti-1}_{tj}",
                f"{ti}_{tj}",
                f"{ti+1}_{tj}",
                f"{ti-1}_{tj-1}",
                f"{ti}_{tj-1}",
                f"{ti+1}_{tj-1}",
                ]
    
    matching_files1 = []
    matching_files2 = []
    
    for filename in os.listdir(path):

        for ti_tj in ti_tj_ls1:

            # Check if 5x5 ti_tj is in filename
            if ti_tj in filename:
                matching_files1.append(os.path.join(path, filename))

        for ti_tj in ti_tj_ls2:

            # Check if 3x3 ti_tj is in filename
            if ti_tj in filename:
                matching_files2.append(os.path.join(path, filename))
    
    return [matching_files1, matching_files2]

################
# extracting overlapping tile extents

# function to get the relevant parallel border of two intersecting tiles i.e. two parallel buffered lines (buffer and tile layer need to be filtered already)
def get_relevant_border(layer_tile, layer_buffer):
    '''
    Function to select only the relevant parallel border to avoid selecting small fields at corners 
    when checking the intersection at the field edges.
    
    Input:
        layer_tile = ogr layer; layer with tile boundaries already filtered for two adjacent tiles
        buffer_tile = ogr layer; layer with buffered tile boundaries already filtered for two adjacent tiles
        
    Output:
        difference_geoms = list; two parallel geometries (lines) to use to find large fields intersecting both boundaries
        
    '''
    # Initialize a variable to hold the unioned geometry
    unioned_geom = None
    
    # Loop through all features in the layer
    for feature in layer_tile:

        geom = feature.GetGeometryRef()

        if unioned_geom is None:
            # If this is the first geometry, initialize unioned_geom with it
            unioned_geom = geom.Clone()

        else:
            # Union the current geometry with the accumulated unioned_geom
            unioned_geom = unioned_geom.Union(geom)
    
    # Get the outer ring of the polygon
    outer_ring = unioned_geom.GetGeometryRef(0)

    # Convert the outer ring into a line geometry
    line = ogr.Geometry(ogr.wkbLineString)

    for n in range(outer_ring.GetPointCount()):

        pt = outer_ring.GetPoint(n)
        line.AddPoint(pt[0], pt[1])
        
    # buffer line
    # set buffer a little bigger than original buffered line to ensure that irrelevant borders are completely removed
    buffer_width = 0.000013 # original: 0.00001°
    line_buffered = line.Buffer(buffer_width)
    
    difference_geoms = []

    # remove overlapping parts
    for feature in layer_buffer:

        geom = feature.GetGeometryRef()
        difference_geom = geom.Difference(line_buffered)
        difference_geoms.append(difference_geom)
    
    return difference_geoms

# get only the relevant border area
def get_border_area(tile_lyr, filter_ls, all_four_borders=False, corner=False):
    '''
    Function to get the relevant border area of adjacent tiles.
    Input:
        tile_lyr = ogr Layer
        filter_ls = list of filter to apply to tile_lyr to select adjacent tiles

    Output:
        all_four_borders = True: unioned geometry of all four borders (with hole)
        corner = True: geometry of the corner of overlaping tiles
        otherwise: intersection 
    '''

    intersections = []

    for f in filter_ls:

        tile_lyr.SetAttributeFilter(f)

        intersection = None

        for tile in tile_lyr:

            geom = tile.GetGeometryRef()

            if intersection is None:
                intersection = geom.Clone()

            else:
                intersection = intersection.Intersection(geom)
                intersections.append(intersection)

    if corner:

        intersection_corner = None

        for geom in intersections:

            if intersection_corner is None:
                intersection_corner = geom.Clone()

            else:
                intersection_corner = intersection_corner.Intersection(geom)

        return intersection_corner

    elif all_four_borders:

        # convert geometries to shapely format to use unary_union function to keep ring shape when dissolving boundary areas
        intersections_shapely = []

        for geom in intersections:
            wkb = geom.ExportToWkb()
            intersections_shapely.append(loads(bytes(wkb)))

        union = unary_union(intersections_shapely)

        # convert shapely geometry back into ogr format
        union_wkb = union.wkb
        union_ogr = ogr.CreateGeometryFromWkb(union_wkb)

        return union_ogr

    else:
        return intersection

#####################
# checking intersections

# Function to check if two geometries truly intersect
def geometries_intersect(geom1, geom2):
    '''
    Function to check if two geometries truly intersect (do not touch)
    
    Input: Two geometries to check.
    
    Output: Boolean True or False
    '''

    return geom1.Intersects(geom2) and not geom1.Touches(geom2)

# Function to check if two bounding boxes intersect
def bbox_intersects(bbox1, bbox2):
    '''
    Function to check if two bounding boxes intersect.

    Input: Two bounding boxes

    Output: Boolean True or False
    '''
    return not (bbox1[1] < bbox2[0] or bbox1[0] > bbox2[1] or
                bbox1[3] < bbox2[2] or bbox1[2] > bbox2[3])

# function to group intersecting features
def group_intersecting_features_from_list(features):
    """
    Group intersecting features in a list using bounding box filtering,
    skipping features with the same "tile_id".

    Input:
        features: List of ogr.Feature objects.

    Output:
        groups: List of sets, where each set contains intersecting features.
    """
    # Initialize a list to store the intersecting groups
    intersecting_groups = []
    
    # Track visited features
    visited = set()
    
    # Get bounding boxes and tile_ids for each feature
    feature_geometries = [feature.GetGeometryRef() for feature in features]
    bounding_boxes = [geom.GetEnvelope() for geom in feature_geometries]
    areas = [geom.GetArea() for geom in feature_geometries]

    # Create list of indices sorted by area in descending order
    # Ensures that large fields are processed first to gather all intersecting fields in one group
    sorted_indices = sorted(range(len(features)), key=lambda i: -areas[i])

    # Function to check if the centroid of the smaller geom is contained
    # in larger geom or the overlap area exceeds a given threshold between 0 and 1 
    # to ensure that only geoms are grouped that truly overlap
    def check_significant_overlap(geom1, geom2, threshold):
        area1 = geom1.GetArea()
        area2 = geom2.GetArea()
        
        if area1 < area2:
            smaller = geom1
            larger = geom2
        else:
            smaller = geom2
            larger = geom1

        intersection_area = geom1.Intersection(geom2).GetArea()
        overlap1 = (intersection_area / area1)
        overlap2 = (intersection_area / area2)

        condition1 = larger.Contains(smaller.Centroid())
        condition2 = overlap1 >= threshold or overlap2 >= threshold

        return condition1 or condition2

    # Function to expand the current group by checking intersections with all features
    def expand_group(current_group, feature_indices):
        stack = list(feature_indices)
        while stack:
            index = stack.pop()
            if index not in visited:
                visited.add(index)
                current_group.add(features[index])
                
                geom1 = feature_geometries[index]
                bbox1 = bounding_boxes[index]

                for j in sorted_indices:
                    if j != index and j not in visited:
                        bbox2 = bounding_boxes[j]
                        if bbox_intersects(bbox1, bbox2):
                            geom2 = feature_geometries[j]
                            if geometries_intersect(geom1, geom2):
                                if check_significant_overlap(geom1, geom2, 0.3):
                                    stack.append(j)

    # Find and group intersections
    for i in sorted_indices:
        if i not in visited:
            current_group = set()
            expand_group(current_group, [i])
            intersecting_groups.append(current_group)
    
    return intersecting_groups

###################
# writing to file

# function to combine cleaned tiles into larger tile
def union_to_large_tile(tile_lyr, core_paths, epsg):
    '''
    Function to create one large tile out of multiple smaller tiles.

    Input:
        tile_lyr = ogr layer
        core_paths = list of paths that correspond to tiles that should be unioned

    Output:
        geometry of unioned tile
    '''

    tile_ids = []

    for path in core_paths:
        file_name = path.split("/")[-1]
        ti, tj = file_name.split("_")[1:3]
        tile_id = f"{epsg}_{ti}_{tj}"
        tile_ids.append(tile_id)

    tiles = []

    for tile_id in tile_ids:
        tile_lyr.SetAttributeFilter(f"tile_id = '{tile_id}'")
        #print(tile_ids)
        tile = tile_lyr.GetNextFeature()
        #print(tile)
        tile_geom = tile.GetGeometryRef()
        tiles.append(tile_geom.Clone())

    union = None

    for tile_geom in tiles:
        if union is None:
            union = tile_geom
        else:
            union = union.Union(tile_geom)

    return union

# function to add water intersection attribute
def add_water_attribute(intersect_features, water_geoms):
    '''
    Function to add water attribute "intersects_water" to features.

    Input:
        list of features
        list of water geometries

    Output:
        list of features with added "intersects_water" attribute
    '''

    water_output = []

    for ifeat in intersect_features: #PR: adjusted to ifeat

        # Add the intersects_water field to the feature definition if it doesn't exist
        layer_defn = ifeat.GetDefnRef()
        if layer_defn.GetFieldIndex("intersects_water") == -1:  # Check if field already exists 
            field_defn = ogr.FieldDefn("intersects_water", ogr.OFTInteger) 
            layer_defn.AddFieldDefn(field_defn) 

        igeom = ifeat.GetGeometryRef() #PR: adjusted to ifeat
        bbox1 = igeom.GetEnvelope() #PR: adjusted to igeom
        intersects = False

        for water_geom in water_geoms:
            bbox2 = water_geom.GetEnvelope()

            if bbox_intersects(bbox1, bbox2):
                if water_geom.Intersects(igeom): #PR: adjusted to igeom
                    intersects = True
                    break

        if intersects:
            ifeat.SetField("intersects_water", 1) #PR: adjusted to ifeat
            water_output.append(feat.Clone())
        else:
            ifeat.SetField("intersects_water", 0) #PR: adjusted to ifeat
            water_output.append(feat.Clone())

    return water_output

def save_gpkg_final(lists, out_path, template_path, lyr_name):
    '''
    Function to save a list of a list of features as gpkg.
    
    Input:
        lists = list; contains several list of features
        out_path = path; defining output gpkg file
        template_path = path; layer template to copy layer definition and EPSG
        lyr_name = string; defining layer name

    Output:
        no returned output; gpkg is saved at out_path location
    '''
    
    driver_out = gdal.GetDriverByName("GPKG")
    driver_in = ogr.GetDriverByName("GPKG")
    
    in_ds = driver_in.Open(template_path)
    in_layer = in_ds.GetLayer()
    in_CRS = in_layer.GetSpatialRef()

    out_ds = driver_out.Create(out_path, 0, 0, 0, gdal.GDT_Unknown)
    out_layer = out_ds.CreateLayer(lyr_name, in_CRS, geom_type=ogr.wkbUnknown)

    # Add fields from original gpkg_layer to the new layer
    layer_defn = in_layer.GetLayerDefn()

    for i in range(layer_defn.GetFieldCount()):

        field_defn = layer_defn.GetFieldDefn(i)
        out_layer.CreateField(field_defn)

    # add intersects_water field to layer
    water_field = ogr.FieldDefn("intersects_water", ogr.OFTInteger)
    out_layer.CreateField(water_field)

    # Add features to the new layer
    for ls in lists:

        if ls == lists[0]:

            fid_ls = range(1, len(ls) + 1)

            for i, feature in enumerate(ls):
                '''
                # FIXME new
                if feature.GetFieldIndex("intersects_water") == -1:
                    feature.SetField("intersects_water", 0)
                '''
                # set FID
                feature.SetFID(fid_ls[i])
                out_layer.CreateFeature(feature)

        else:
            last_fid = fid_ls[-1]
            fid_ls = range(last_fid + 1, len(ls) + 1)

            for i, feature in enumerate(ls):
                # set FID
                feature.SetFID(fid_ls[i])
                out_layer.CreateFeature(feature)

    # clean up
    out_ds = None
    out_layer = None
    time.sleep(1)
    in_ds = None
    in_layer = None

# Function to clean up features to delete afterwards
def cleanup_features(features):
    for feat in features:
        feat = None