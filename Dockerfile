FROM geodata/gdal:2.1.3

# http://lifeonubuntu.com/ubuntu-missing-add-apt-repository-command/
# RUN apt-get update && apt-get install -y software-properties-common
RUN apt-get update && apt-get install -y curl unzip
# Use development versions for newer GDAL (2+) and other geo libraries
# RUN add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable
# RUN apt-get update && apt-get install -y unzip gdal-bin libgeos-dev libproj-dev
