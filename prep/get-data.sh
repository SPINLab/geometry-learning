#!/usr/bin/env bash
set -ex
mkdir -p /data/files
cd /data/files

# Get the data
curl -o TOP10NL_25W.zip https://geodata.nationaalgeoregister.nl/top10nlv2/extract/kaartbladen/TOP10NL_25W.zip?formaat=gml
curl -o TOP10NL_34O.zip https://geodata.nationaalgeoregister.nl/top10nlv2/extract/kaartbladen/TOP10NL_34O.zip?formaat=gml
curl -o netherlands-latest-free.shp.zip http://download.geofabrik.de/europe/netherlands-latest-free.shp.zip

# Inflate
unzip -o TOP10NL_25W.zip
unzip -o TOP10NL_34O.zip
unzip -o netherlands-latest-free.shp.zip *buildings*

# Load the database. Be sure to have the postgis container running
ogr2ogr -f "PostgreSQL" PG:"host=postgis port=5432 dbname=postgres user=postgres password=postgres" TOP10NL_25W.gml -overwrite -progress -t_srs "EPSG:4326" -oo GML_ATTRIBUTES_TO_OGR_FIELDS=YES
ogr2ogr -f "PostgreSQL" PG:"host=postgis port=5432 dbname=postgres user=postgres password=postgres" TOP10NL_34O.gml -append -progress -t_srs "EPSG:4326" -oo GML_ATTRIBUTES_TO_OGR_FIELDS=YES
# https://trac.osgeo.org/gdal/ticket/4939
# http://www.bostongis.com/PrinterFriendly.aspx?content_name=ogr_cheatsheet
ogr2ogr -f "PostgreSQL" PG:"host=postgis port=5432 dbname=postgres user=postgres password=postgres" gis.osm_buildings_a_free_1.shp -overwrite -progress -nln osm_buildings -nlt PROMOTE_TO_MULTI -lco EXTRACT_SCHEMA_FROM_LAYER_NAME=no

bash ./export-data.sh