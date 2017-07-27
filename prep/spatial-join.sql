SELECT st_astext(st_snaptogrid(gebouw.wkb_geometry, 0.000001)) AS brt_wkt,
  st_astext(st_snaptogrid(osm_buildings.wkb_geometry, 0.000001)) AS osm_wkt,
  st_astext(st_snaptogrid(st_intersection(gebouw.wkb_geometry, osm_buildings.wkb_geometry), 0.000001)) AS intersection_wkt
FROM gebouw, osm_buildings
WHERE st_intersects(st_buffer(gebouw.wkb_geometry, 0.00005), osm_buildings.wkb_geometry) AND
	st_issimple(st_snaptogrid(gebouw.wkb_geometry, 0.000001)) AND
	st_issimple(st_snaptogrid(osm_buildings.wkb_geometry, 0.000001)) AND
	st_issimple(st_snaptogrid(st_intersection(gebouw.wkb_geometry, osm_buildings.wkb_geometry), 0.000001))
LIMIT 100000;
