#!/usr/bin/env bash
rm /data/files/topology-training.csv
# extract the joined data
# https://gis.stackexchange.com/questions/185072/ogr2ogr-sql-query-from-text-file#185141
cd /data/files
set -ex
ogr2ogr -f CSV topology-training.csv PG:"host=postgis port=5432 dbname=postgres user=postgres password=postgres" -sql @../prep/spatial-join.sql
lines=$(tail -n +2 topology-training.csv | wc -l)
echo
echo "Wrote $lines number of data points"
echo "The script ran successfully. The generated data set was saved to files/topology-training.csv"
