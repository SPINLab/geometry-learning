from topoml_util.geom_loss import r4_bivariate_gaussian
from keras import backend as K


class GaussianMixtureLoss:
    def __init__(self, num_components):
        self.num_components = num_components

    def geom_gaussian_mixture_loss(self, y_true, y_pred):
        # loss fn based on eq #26 of http://arxiv.org/abs/1308.0850.
        (data_points, points, features) = y_pred.shape
        geom_type_index = 6 * self.num_components  # Calculate offset from parameters times components
        render_index = geom_type_index + 8
        pi_index = 5

        predicted_components = K.reshape(y_pred[:geom_type_index], (-1, points.value, self.num_components, 6))
        pi_weights = K.softmax(predicted_components[:, :, :, pi_index])

        true_components = K.reshape(y_true[:geom_type_index], (-1, points.value, self.num_components, 6))

        gmm = r4_bivariate_gaussian(true_components, predicted_components) * pi_weights
        gmm_loss = K.log(K.sum(-K.log(gmm + K.epsilon()), keepdims=True))

        # Zero out loss terms beyond N_s, the last actual stroke
        # render = 1 - K.mean(y_pred[:, :, render_index:render_index + 2])  # RENDER and STOP values
        # gmm_loss = gmm_loss * render

        geom_type_error = K.categorical_crossentropy(
            K.softmax(y_true[:, :, geom_type_index:render_index]),
            K.softmax(y_pred[:, :, geom_type_index:render_index]))
        render_error = K.categorical_crossentropy(
            K.softmax(y_true[:, :, render_index:]),
            K.softmax(y_pred[:, :, render_index:]))

        return gmm_loss + geom_type_error + render_error
