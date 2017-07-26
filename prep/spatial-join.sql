SELECT st_astext(gebouw.wkb_geometry) AS source_wkt, st_astext(osm_buildings.wkb_geometry) AS target_wkt, st_astext(st_intersection(gebouw.wkb_geometry, osm_buildings.wkb_geometry)) AS intersection_wkt
FROM gebouw, osm_buildings
WHERE st_intersects(st_buffer(gebouw.wkb_geometry, 0.00005), osm_buildings.wkb_geometry);
