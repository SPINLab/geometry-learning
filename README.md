# Topology-Learning
A machine learning project for learning geospatial topology

# Installation
If you value your time, go for the dockerized version.

## Docker

Due to the cross-platform dependency issues installing python geospatial packages, the setup has been [Dockerized](https://www.docker.com/) to ensure cross-platform interoperability. This is the easiest way to use. 
 

- See [instructions](https://docs.docker.com/engine/installation/#supported-platforms) on installing and running Docker on your platform.
- Install [docker compose](https://docs.docker.com/compose/install/)
- Run `docker-compose build` in the root folder of this repository
- **!! Make sure `./script/get-data.sh` has LF line endings if you run this on windows!!**. To be sure, you can install [Git for windows](https://git-for-windows.github.io/), open a git bash shell in the repo folder and issue a `dos2unix ./script/get-data.sh`. This will convert the line endings to unix line feeds.
- Run `docker-compose up`. Once the container has exited, you should see a `files` directory in your repo folder.

## Using self-installed libraries
If you need to run or debug locally, you need the following dependencies:

|Package|Linux|Mac|Windows|
|:------|:----|:---|:------|
|Shapely|`apt-get install -y libgeos-dev && pip install shapely`| |download and `pip install` [one of these wheels](http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely)|
|GDAL|`apt-get install -y libproj-dev && git clone https://github.com/OSGeo/gdal.git && cd gdal/gdal && ./configure --with-python && make >/dev/null 2>&1 && make install && ldconfig`| |download and `pip install` [one of these wheels](http://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)|

