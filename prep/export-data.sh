#!/usr/bin/env bash
echo "exporting data, this will take a couple of minutes..."
rm /data/files/topology-training.csv
# extract the joined data
# https://gis.stackexchange.com/questions/185072/ogr2ogr-sql-query-from-text-file#185141
cd /data/files
set -ex
ogr2ogr -f CSV brt_osm.csv PG:"host=postgis port=5432 dbname=postgres user=postgres password=postgres" -sql @../prep/spatial-join.sql
set -e
lines=$(tail -n +2 brt_osm.csv | wc -l)
echo
echo "Wrote $lines number of data points"
echo "The export script ran successfully. The generated data set was saved to files/brt_osm.csv"

cd ../prep
echo "Creating BRT/OSM numpy archive..."
python3 vectorize_brt_osm.py
echo "Creating neighborhoods numpy archive..."
python3 get-neighborhoods.py

echo "Done!"