from matplotlib.collections import PatchCollection
from shapely import wkt
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon


def wkt2pyplot(input_wkts, target_wkts, prediction_wkts, input_color='green', target_color='red', pred_color='blue'):
    """
    Convert input, target and prediction well-known encoded geometry arrays to pyplot
    :param input_wkts: an array of input geometries, rendered in (standard) green
    :param target_wkts: an array of target geometries, rendered in (standard) red
    :param prediction_wkts: an array of prediction geometries, rendered in (standard) blue
    :param input_color: a pyplot-compatible notation of color
    :param pred_color: a pyplot-compatible notation of color
    :param target_color: a pyplot-compatible notation of color
    :return: a matplotlib pyplot
    """
    input_geoms = [wkt.loads(input_wkt) for input_wkt in input_wkts]
    target_geoms = [wkt.loads(target_wkt) for target_wkt in target_wkts]
    prediction_geoms = [wkt.loads(prediction_wkt) for prediction_wkt in prediction_wkts]

    for input_geom in input_geoms:
        if not input_geom.boundary:
            raise ValueError("Don't know how to plot input geom", input_geom)

    fig, ax = plt.subplots()

    input_polys = [Polygon(input_geom.boundary.coords) for input_geom in input_geoms]

    inputs = PatchCollection(input_polys, alpha=0.4, linewidth=1)
    inputs.set_color(input_color)
    ax.add_collection(inputs)

    # target_polys = [Polygon(target_geom.boundary.coords) for target_geom in target_geoms]
    # targets = PatchCollection(target_polys, alpha=0.4, linewidth=1)
    # targets.set_color(target_color)
    # ax.add_collection(targets)

    plt.plot([geom.coords.xy[0][0] for geom in target_geoms],
             [geom.coords.xy[1][0] for geom in target_geoms],
             marker='o', color=pred_color, linewidth=0)

    plt.plot([geom.coords.xy[0][0] for geom in prediction_geoms],
             [geom.coords.xy[1][0] for geom in prediction_geoms],
             marker='o', color=target_color, linewidth=0)
    plt.axis('auto')

    return plt
