FROM geodata/gdal:2.1.3

RUN apt-get update && apt-get install -y curl unzip
RUN pip install shapely numpy
WORKDIR /data/prep