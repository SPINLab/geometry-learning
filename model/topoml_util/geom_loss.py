from keras import backend as K
from .GeoVectorizer import GEOM_TYPE_INDEX, RENDER_INDEX


def geom_loss(y_true, y_pred):
    print(y_true.shape)
    coordinate_error = K.log(y_true[:, :, 0:2]) - K.log(y_pred[:, :, 0:2])
    geom_type_error = K.softmax(y_true[:, :, GEOM_TYPE_INDEX:RENDER_INDEX]) - K.softmax(y_pred[:, :, GEOM_TYPE_INDEX:RENDER_INDEX])
    render_error = K.softmax(y_true[:, :, RENDER_INDEX:]) - K.softmax(y_pred[:, :, RENDER_INDEX:])
    reconstructed = K.concatenate([coordinate_error, geom_type_error, render_error], axis=2)
    return K.sum(reconstructed)
