FROM geodata/gdal:2.1.3

RUN apt-get update && apt-get install -y curl unzip