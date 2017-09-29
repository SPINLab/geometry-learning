from topoml_util.geom_loss import r4_bivariate_gaussian, r3_bivariate_gaussian, r3_univariate_gaussian
from keras import backend as K


class GaussianMixtureLoss:
    def __init__(self, num_components, num_points):
        self.num_points = num_points
        self.num_components = num_components

    def geom_gaussian_mixture_loss(self, y_true, y_pred):
        """
        Calculates a loss from a rank 3 sequence, representing a self.num_components * 6 slice (the mixture components)
        plus a
        :param y_true: rank 3 of shape(records, points, point features) truth values tensor
        :param y_pred: rank 3 of shape(records, points, point features) predicted values tensor
        :return: a summed mixture loss and categorical cross entropy losses for the geometry type and stop bits
        """
        # loss fn based on eq #26 of http://arxiv.org/abs/1308.0850.
        geom_type_index = 6 * self.num_components  # Calculate offset from parameters times components
        render_index = geom_type_index + 8

        true_components = y_true[:, :, :geom_type_index]
        true_components = K.reshape(true_components, (-1, self.num_points, self.num_components, 6))
        predicted_components = K.reshape(y_pred[:, :, :geom_type_index], (-1, self.num_points, self.num_components, 6))

        pi_index = 5
        pi_weights = K.softmax(predicted_components[:, :, :, pi_index])
        gmm = r4_bivariate_gaussian(true_components, predicted_components) * pi_weights
        gmm_loss = K.log(K.sum(-K.log(gmm + K.epsilon()), keepdims=True))

        # TODO: Zero out loss terms beyond the last point
        # render = 1 - K.mean(y_pred[:, :, render_index:render_index + 2])  # RENDER and STOP values
        # gmm_loss = gmm_loss * render

        geom_type_error = K.categorical_crossentropy(
            K.softmax(y_true[:, :, geom_type_index:render_index]),
            K.softmax(y_pred[:, :, geom_type_index:render_index]))
        render_error = K.categorical_crossentropy(
            K.softmax(y_true[:, :, render_index:]),
            K.softmax(y_pred[:, :, render_index:]))

        return gmm_loss + geom_type_error + render_error

    def r3_univariate_gmm_loss(self, y_true, y_pred):
        """
        A simple loss function for rank 2 single gaussian mixture models
        :param y_true: rank 2 of shape(records, record features) truth values tensor
        :param y_pred: rank 2 of shape(records, record features) truth values tensor
        :return: rank 3 loss values tensor
        """
        # true_components = K.reshape(y_true, (-1, self.num_components, 3))
        # predicted_components = K.reshape(y_pred, (-1, self.num_components, 3))

        # pi_index = 2
        # pi_weights = K.softmax(predicted_components[:, :, pi_index])
        gmm = r3_univariate_gaussian(y_true, y_pred)
        gmm_loss = -K.log(gmm + K.epsilon())

        return gmm_loss
