SELECT st_astext(st_snaptogrid(gebouw.wkb_geometry, 0.0000001)) AS brt_wkt,
  st_astext(st_snaptogrid(ST_GeometryN(osm_buildings.wkb_geometry, 1), 0.0000001)) AS osm_wkt,
  st_astext(st_snaptogrid(st_intersection(gebouw.wkb_geometry, osm_buildings.wkb_geometry), 0.0000001)) AS intersection_wkt,
  st_distance(st_transform(st_centroid(gebouw.wkb_geometry), 28992), st_transform(st_centroid(osm_buildings.wkb_geometry), 28992))::real AS centroid_distance,
  st_distance(st_transform(gebouw.wkb_geometry, 28992), st_transform(osm_buildings.wkb_geometry, 28992))::real AS geom_distance,
  st_astext(st_snaptogrid(st_centroid(gebouw.wkb_geometry), 0.0000001)) AS brt_centroid_wkt,
  st_astext(st_snaptogrid(st_centroid(osm_buildings.wkb_geometry), 0.0000001)) AS osm_centroid_wkt,
  st_astext(st_snaptogrid(st_transform(st_centroid(gebouw.wkb_geometry), 28992), 0.0000001)) AS brt_centroid_wkt_rd,
  st_astext(st_snaptogrid(st_transform(st_centroid(osm_buildings.wkb_geometry), 28992), 0.0000001)) AS osm_centroid_wkt_rd,
  st_area(st_transform(st_intersection(gebouw.wkb_geometry, osm_buildings.wkb_geometry), 28992))::real AS intersection_surface
FROM gebouw, osm_buildings
WHERE
  -- Allow only polygons (there are a few point buildings in there, don't ask me why)
	ST_GeometryType(gebouw.wkb_geometry) = 'ST_Polygon' AND
	-- Expand each source geometry with a buffer of a few meters to include non-intersecting target geometries
	st_intersects(st_buffer(gebouw.wkb_geometry, 0.00005), osm_buildings.wkb_geometry) AND
	-- Guarantee good geometries
	st_issimple(st_snaptogrid(gebouw.wkb_geometry, 0.000001)) AND
	st_issimple(st_snaptogrid(osm_buildings.wkb_geometry, 0.000001)) AND
	st_issimple(st_snaptogrid(st_intersection(gebouw.wkb_geometry, osm_buildings.wkb_geometry), 0.000001)) AND
	-- Restrict to ringless polygons
	ST_NumInteriorRings(gebouw.wkb_geometry) = 0 AND
	ST_NumInteriorRings(ST_GeometryN(osm_buildings.wkb_geometry, 1)) = 0
LIMIT 500000