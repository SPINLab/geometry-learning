from topoml_util.geom_loss import bivariate_gaussian, univariate_gaussian
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
        # TODO: make rank agnostic
        tc_shape = y_true.shape
        if not tc_shape.ndims == 3:
            raise ValueError('This function works on tensors of rank 3.')

        # loss fn based on eq #26 of http://arxiv.org/abs/1308.0850.
        geom_type_index = 6 * self.num_components  # Calculate offset from parameters times components
        render_index = geom_type_index + 8

        true_components = y_true[..., :geom_type_index]
        shape = [-1, self.num_points, self.num_components, 6]
        true_components = K.reshape(true_components, tuple(shape))

        # TODO: make reshape op rank agnostic
        predicted_components = K.reshape(y_pred[..., :geom_type_index],
                                         (-1, self.num_points, self.num_components, 6))

        pi_index = 5  # mixture component weight
        pi_weights = K.softmax(predicted_components[..., pi_index])
        gmm = bivariate_gaussian(true_components, predicted_components) * pi_weights
        gmm_loss = K.log(K.sum(-K.log(gmm + K.epsilon())))

        # TODO: Zero out loss terms beyond the last point
        # render = 1 - K.mean(y_pred[:, :, render_index:render_index + 2])  # RENDER and STOP values
        # gmm_loss = gmm_loss * render

        geom_type_error = K.categorical_crossentropy(
            K.softmax(y_true[..., geom_type_index:render_index]),
            K.softmax(y_pred[..., geom_type_index:render_index]))
        render_error = K.categorical_crossentropy(
            K.softmax(y_true[..., render_index:]),
            K.softmax(y_pred[..., render_index:]))

        return gmm_loss + geom_type_error + render_error

    def univariate_gmm_loss(self, true, pred):
        """
        A simple loss function for rank 3 single gaussian mixture models
        :param true: truth values tensor
        :param pred: prediction values tensor
        :return: loss values tensor
        """
        if not true.shape == pred.shape:
            print(
                'Warning: truth', true.shape, 'and prediction tensors', pred.shape, 'do not have the same shape. The '
                'outcome of the loss function may be unpredictable.')

        # true_components = K.reshape(true, (-1, self.num_components, 3))
        # TODO: make reshape op rank agnostic
        predicted_components = K.reshape(pred, (-1, self.num_components, 3))

        pi_index = 2
        pi_weights = K.softmax(pred[..., pi_index])
        gmm = univariate_gaussian(true, predicted_components) * pi_weights
        gmm_loss = -K.log(K.sum(gmm + K.epsilon()))

        return gmm_loss
