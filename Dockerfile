FROM python:3.6

RUN apt-get update && apt-get install -y unzip libgeos-dev libproj-dev
RUN pip3 install numpy shapely

# Build GDAL
RUN git clone https://github.com/OSGeo/gdal.git
WORKDIR /gdal/gdal
RUN ./configure --with-python && \
  echo 'Building GDAL, this may take an hour' && \
  make >/dev/null 2>&1 && \
  make install && \
  ldconfig