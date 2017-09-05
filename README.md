# Topology-Learning
A machine learning project for learning geospatial topology

# Data preparation
If you value your time, go for the dockerized version.
Note there is a pre-built `geodata-vectorized.npz` under `files`, so you don't need to rebuild the training data.

## Numpy archive description
The numpy archive contains vectors deserialized from well-known text geometries, using the [shapely](https://pypi.python.org/pypi/Shapely) library.

### input_geoms
A nested array of shape (?, ?, 16) - depending on the sequence length settings - of deserialized WKT polygons with the following encoding:
- [:, :, 0:2]: The polygon point/node coordinates in lon/lat
- [:, :, 2:5]: A one-hot encoded sequence of actions: 

# intersection
# centroid_distance
# geom_distance
# brt_centroid
# osm_centroid
# centroids
# centroids_rd


## Docker
Due to the cross-platform dependency issues installing python geospatial packages, the setup has been [Dockerized](https://www.docker.com/) to ensure cross-platform interoperability. This is the easiest way to use. 

- See [instructions](https://docs.docker.com/engine/installation/#supported-platforms) on installing and running Docker on your platform.
- Install [docker compose](https://docs.docker.com/compose/install/)
- Run `docker-compose build` in the root folder of this repository
- **!! Make sure `./script/get-data.sh` has LF line endings if you run this on windows!!**. To be sure, you can install [Git for windows](https://git-for-windows.github.io/), open a git bash shell in the repo folder and issue a `dos2unix ./script/*.sh`. This will convert the line endings to unix line feeds.
- Run `docker-compose up`. Once the container has exited, you should see a `files` directory in your repo folder with a file `topology-training.csv`. This contains your extracted geometries, distances and intersections. 

## Using self-installed libraries
If you need to run or debug locally, you need the following dependencies:

|Package|Linux|Mac|Windows|
|:------|:----|:---|:------|
|Shapely|`apt-get install -y libgeos-dev && pip install shapely`| |download and `pip install` [one of these wheels](http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely)|
|GDAL|`apt-get install -y libproj-dev && git clone https://github.com/OSGeo/gdal.git && cd gdal/gdal && ./configure --with-python && make >/dev/null 2>&1 && make install && ldconfig`| |download and `pip install` [one of these wheels](http://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)|
|PostGIS|`apt-get install postgres postgis` | `brew install -y postgres postgis`| see [here](http://postgis.net/windows_downloads/) |

Once you have everything installed, cd to the prep directory and execute `get-data.sh` en `export-data.sh`.

## Conversion to numpy archive
After you have created your `topology-training.csv`, convert it to a numpy archive using `cd model && python3 vectorize.py`. Note there is a pre-built `geodata-vectorized.npz` under `files`

# Running the models
Models are run from the `model` directory. Each one of them is a more or less elaborate strategy to work with real-valued data, often combined with categorical data, using LSTMs.

You can monitor progress using tensorboard:
`tensorboard --log_dir ./model/tensorboard`
or
`python3 -m tensorflow.tensorboard --logdir ./model/tensorboard`

## Intersection auto-encoder
Issue `python3 intersection.py`. Approximates the intersection of two geometries into a new geometry. Work in progress.

## Fixed 1d gaussian
The most simple of gaussian approximations. Attempts to approximate a mu of 52. as close as possible with a sigma of 0. Converges in about 10 epochs to exactly [52. 0.].
`python3 fixed_1d_gaussian.py`

## Fixed 2d gaussian
Attempts to approximate a bivariate gaussian with mu's of a realistic coordinate set of [5, 52], sigmas of 0 and rho of 0. Work in progress.
`python3 fixed_2d_gaussian.py`

## 'Geometric' distance
Approximates the distance between two polygons in meters, expressed as a single gaussian. This is a vanilla distance query approximation. If the polygons intersect, the distance is 0. Work in progress.
`python3 geom_distance.py`

## 'Centroidal' distance
Calculate the distance between the centroids (of two polygons) in meters, expressed as a single gaussian. Converges to deviations of centimeters in about 50 epochs.
`python3 centroid_distance.py`

## Character-level model
There is one model that uses a character-level strategy to approximate intersections. Work in progress.
