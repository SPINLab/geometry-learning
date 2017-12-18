#!/usr/bin/env bash
echo "importing data... this will take a while, depending on your internet connection speed"
set -ex
mkdir -p /data/files
cd /data/files

# Get the Base Registration for Topograpy data
curl -o base_registration_topography/TOP10NL_25W.zip https://geodata.nationaalgeoregister.nl/top10nlv2/extract/kaartbladen/TOP10NL_25W.zip?formaat=gml
curl -o base_registration_topography/TOP10NL_34O.zip https://geodata.nationaalgeoregister.nl/top10nlv2/extract/kaartbladen/TOP10NL_34O.zip?formaat=gml

# Get the OpenStreetMap data
curl -o openstreetmap/netherlands-latest-free.shp.zip http://download.geofabrik.de/europe/netherlands-latest-free.shp.zip

# Get neighborhoods
curl -X GET \
  -o neighborhoods/neighborhoods.csv \
  'https://geodata.nationaalgeoregister.nl/wijkenbuurten2017/wfs?request=GetFeature&service=WFS&version=2.0.0&typeName=cbs_buurten_2017&outputFormat=csv&srsName=EPSG%3A4326&PropertyName=aantal_inwoners%2Cgeom' \
  -H 'cache-control: no-cache'

# Get BAG buildings
woonfunctie
winkelfunctie
bijeenkomstfunctie
onderwijsfunctie
gezondheidsfunctie
kantoorfunctie
industriefuntie
sportfunctie
logiesfunctie

# Inflate
unzip -o base_registration_topography/TOP10NL_25W.zip
unzip -o base_registration_topography/TOP10NL_34O.zip
unzip -o openstreetmap/netherlands-latest-free.shp.zip *buildings*

# Load the database. Be sure to have the postgis container running
ogr2ogr -f "PostgreSQL" PG:"host=postgis port=5432 dbname=postgres user=postgres password=postgres" base_registration_topography/TOP10NL_25W.gml -overwrite -progress -t_srs "EPSG:4326" -oo GML_ATTRIBUTES_TO_OGR_FIELDS=YES
ogr2ogr -f "PostgreSQL" PG:"host=postgis port=5432 dbname=postgres user=postgres password=postgres" base_registration_topography/TOP10NL_34O.gml -append -progress -t_srs "EPSG:4326" -oo GML_ATTRIBUTES_TO_OGR_FIELDS=YES
# https://trac.osgeo.org/gdal/ticket/4939
# http://www.bostongis.com/PrinterFriendly.aspx?content_name=ogr_cheatsheet
ogr2ogr -f "PostgreSQL" PG:"host=postgis port=5432 dbname=postgres user=postgres password=postgres" openstreetmap/gis.osm_buildings_a_free_1.shp -overwrite -progress -nln osm_buildings -nlt PROMOTE_TO_MULTI -lco EXTRACT_SCHEMA_FROM_LAYER_NAME=no

bash ./export-data.sh