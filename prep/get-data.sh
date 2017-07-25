#!/usr/bin/env bash
set -ex
mkdir -p /data/files
cd /data/files
# Get the data
curl -o TOP10NL_25W.zip http://geodata.nationaalgeoregister.nl/top10nlv2/extract/kaartbladen/TOP10NL_25W.zip?formaat=gml
curl -o TOP10NL_34O.zip http://geodata.nationaalgeoregister.nl/top10nlv2/extract/kaartbladen/TOP10NL_34O.zip?formaat=gml
curl -o netherlands-latest-free.shp.zip http://download.geofabrik.de/europe/netherlands-latest-free.shp.zip

# inflate
unzip -o TOP10NL_25W.zip
unzip -o TOP10NL_34O.zip
unzip -o netherlands-latest-free.shp.zip
