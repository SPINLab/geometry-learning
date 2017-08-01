from shapely import wkt, geometry


class Vectorize:
    @staticmethod
    def vectorize(wkt_geom):
        geo_shape = wkt.loads(wkt_geom)
        if geo_shape.geom_type == "Point":
            coords = list(geo_shape.coords)
            return coords
        elif geo_shape.geom_type == "Polygon":
            outer_ring = geo_shape.exterior
            coords = list(outer_ring.coords)
            return coords
        else:
            return None
