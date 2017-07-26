from osgeo import gdal, osr, ogr

# Enable GDAL/OGR exceptions
gdal.UseExceptions()

print('Loading data...')
amsterdam = gdal.ogr.Open('../files/TOP10NL_25W.gml', 0)
enschede = gdal.ogr.Open('../files/TOP10NL_34O.gml', 0)
osm_buildings = gdal.ogr.Open('../files/gis.osm_buildings_a_free_1.shp', 0)

source_layers = [amsterdam.GetLayer('Gebouw'), enschede.GetLayer('Gebouw')]
target_layers = [osm_buildings.GetLayer()]

print('Processing features...')
for source_layer in source_layers:
    # input SpatialReference
    in_spatial_ref = source_layer.GetSpatialRef()

    # output SpatialReference
    out_spatial_ref = osr.SpatialReference()
    out_spatial_ref.ImportFromEPSG(4326)

    # create the CoordinateTransformation
    coord_trans = osr.CoordinateTransformation(in_spatial_ref, out_spatial_ref)

    in_extent_list = list(source_layer.GetExtent())
    bottom_left = ogr.Geometry(ogr.wkbPoint)
    top_right = ogr.Geometry(ogr.wkbPoint)
    bottom_left.AddPoint_2D(in_extent_list[0], in_extent_list[2])
    top_right.AddPoint_2D(in_extent_list[1], in_extent_list[3])
    bottom_left.Transform(coord_trans)
    top_right.Transform(coord_trans)
    print(bottom_left, top_right)

    for target_layer in target_layers:
        print('Setting filter')
        target_layer.SetSpatialFilterRect(bottom_left.GetX(), bottom_left.GetY(), top_right.GetX(), top_right.GetY())
        print('Filter:', target_layer.GetSpatialFilter())

        print('Iterating over features')
        in_feature = source_layer.GetNextFeature()
        in_feature_counter = 0
        while in_feature:
            in_feature_counter += 1
            in_geom = in_feature.GetGeometryRef()
            in_geom.Transform(coord_trans)

            target_layer.ResetReading()
            out_feature = target_layer.GetNextFeature()
            out_feature_counter = 0
            while out_feature:
                out_feature_counter += 1
                out_geom = out_feature.GetGeometryRef()

                if in_geom.Intersects(out_geom):
                    intersection_wkt = in_geom.Intersection(out_geom).ExportToWkt()
                    print('IN', in_geom, 'OUT', out_geom, 'INTERSECTION', intersection_wkt)

                out_feature = target_layer.GetNextFeature()

            in_feature = source_layer.GetNextFeature()

        print('Processed %s in_features and %s out_features' % (in_feature_counter, out_feature_counter))

print('The data prepping operation completed successfully')
