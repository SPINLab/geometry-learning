from osgeo import osr, gdal


def gdal_error_handler(err_class, err_num, err_msg):
    err_type = {
        gdal.CE_None: 'None',
        gdal.CE_Debug: 'Debug',
        gdal.CE_Warning: 'Warning',
        gdal.CE_Failure: 'Failure',
        gdal.CE_Fatal: 'Fatal'
    }
    err_msg = err_msg.replace('\n', ' ')
    err_class = err_type.get(err_class, 'None')
    print('Error Number: %s' % err_num)
    print('Error Type: %s' % err_class)
    print('Error Message: %s' % err_msg)


# install error handler
gdal.PushErrorHandler(gdal_error_handler)


def layerToWGS(in_layer):
    out_driver = gdal.ogr.GetDriverByName('MEMORY')
    out_dataset = out_driver.CreateDataSource('Output datasource')
    out_layer = out_dataset.CreateLayer('Gebouw', geom_type=in_layer.GetGeomType())

    # input SpatialReference
    in_spatial_ref = osr.SpatialReference()
    in_spatial_ref.ImportFromEPSG(28992)

    # output SpatialReference
    out_spatial_ref = osr.SpatialReference()
    out_spatial_ref.ImportFromEPSG(4326)

    # create the CoordinateTransformation
    coord_trans = osr.CoordinateTransformation(in_spatial_ref, out_spatial_ref)

    in_layer_defn = in_layer.GetLayerDefn()
    # get the output layer's feature definition
    out_layer_defn = out_layer.GetLayerDefn()

    for i in range(0, in_layer_defn.GetFieldCount()):
        field_defn = in_layer_defn.GetFieldDefn(i)
        out_layer.CreateField(field_defn)

    # loop through the input features
    in_feature = in_layer.GetNextFeature()
    while in_feature:
        # get the input geometry
        geometry = in_feature.GetGeometryRef()
        # reproject the geometry
        geometry.Transform(coord_trans)
        # create a new feature
        out_feature = in_feature.Clone()
        # set the geometry and attribute
        out_feature.SetGeometry(geometry)
        # out_feature.SetFieldsFrom(in_feature)
        # for i in range(0, out_layer_defn.GetFieldCount()):
            # out_feature.SetField(out_layer_defn.GetFieldDefn(i).GetNameRef(), in_feature.GetField(i))
        # add the feature to the layer
        out_layer.CreateFeature(out_feature)
        # dereference the features and get the next input feature
        out_feature = None
        in_feature = in_layer.GetNextFeature()

    return out_layer
