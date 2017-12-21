import numpy as np
from pyefd import elliptic_fourier_descriptors


def geom_fourier_descriptors(shapes, order):
    """
    Creates a stacked array of different variations of fourier descriptors: normalized, non-normalized
    :param shapes: a list of shapely shapes
    :param order: the fourier descriptor order (the size of the returned array along the 0-axis)
    :return: a 2d array with shape (order * 2, 4)
    """
    fourier_descriptors = []
    for index, shape in enumerate(shapes):
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
            # coords = np.array(boundary.coords)
            # centroid_distances = [boundary.centroid.distance(Point(point)) for point in coords]
            # centroid_fourier_descriptors = elliptic_fourier_descriptors(centroid_distances, normalize=True)

            # Stack 'em all
            coeffs = np.append(non_normalized_coeffs, normalized_coeffs)  # without axis this will just create an array
            coeffs = np.append(coeffs, [boundary.area, boundary.length, len(boundary.coords)])
            fourier_descriptors.append(coeffs)
        except Exception as e:
            print('Error %s on geom at csv line %i' % (e, index + 2))
            raise e

    return fourier_descriptors
