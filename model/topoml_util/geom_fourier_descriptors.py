import numpy as np
from pyefd import elliptic_fourier_descriptors

np.seterr(all='raise')


def geom_fourier_descriptors(shapes, order):
    """
    Creates a stacked array of different variations of fourier descriptors: normalized, non-normalized
    :param shapes: a list of shapely shapes
    :param order: the fourier descriptor order (the size of the returned array along the 0-axis)
    :return: a 2d array with shape ((order * 2) + 3, 4)
    """
    fourier_descriptors = []
    for index, shape in enumerate(shapes):
        coeffs = create_geom_fourier_descriptor(shape, order)
        fourier_descriptors.append(coeffs)

    return fourier_descriptors


def create_geom_fourier_descriptor(shape, order):
    boundary = shape.boundary
    while boundary.geom_type == "MultiLineString":
        boundary = boundary.geoms[0]
    try:
        # Set normalize to false to retain size information.
        non_normalized_coeffs = elliptic_fourier_descriptors(
            boundary.coords, order=order, normalize=False)
        # normalized Fouriers
        normalized_coeffs = elliptic_fourier_descriptors(
            boundary.coords, order=order, normalize=True)

        # TODO: create centroid distance fourier descriptors
        # See https://doi-org.vu-nl.idm.oclc.org/10.1016/j.image.2009.04.001
        # coords = np.array(boundary.coords)
        # centroid_distances = [boundary.centroid.distance(Point(point)) for point in coords]
        # centroid_fourier_descriptors = elliptic_fourier_descriptors(centroid_distances, normalize=True)

        # Stack 'em all
        coeffs = [shape.area, boundary.length, len(boundary.coords)]
        for nn, n in zip(non_normalized_coeffs, normalized_coeffs):
            coeffs = np.append(coeffs, nn)  # without axis this will just create an array
            coeffs = np.append(coeffs, n)
    except Exception as e:
        print('Error %s on geom ' % e)
        raise e

    return coeffs