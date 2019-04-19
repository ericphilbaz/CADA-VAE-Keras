import keras.backend as K
import numpy as np
from keras.layers import Layer


class KL_W(Layer):

    def __init__(self, dim_stoc, **kwargs):
        self.dim_stoc = dim_stoc
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.anneal = self.add_weight(name='anneal', shape=(), initializer=lambda x: np.array(1.), trainable=False)
        super().build(input_shape)

    def call(self, var_prms):
        """
        We later add (instead of substract) the KL-term, thus the formula represents (-kl)
        """
        mu, logvar = var_prms[:, :self.dim_stoc], var_prms[:, self.dim_stoc:]
        kl = self.anneal * 0.5 * K.sum(K.exp(logvar) + K.square(mu) - 1. - logvar, axis=-1)
        return kl

    def compute_output_shape(self, input_shape):
        return input_shape[:1]

    def get_config(self):
        config = {'dim_stoc': self.dim_stoc}
        base_config = super(KL_W, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class L1Reconstruction(Layer):

    def __init__(self, **kwargs):
        super(L1Reconstruction, self).__init__(**kwargs)

    def build(self, input_shape):
        super(L1Reconstruction, self).build(input_shape)
        self.l1_anneal = self.add_weight(name='l1_weight', shape=(), initializer=lambda x: np.array(1.),
                                         trainable=False)

    def call(self, input_list):
        """ Sum of elementwise absolute differences."""
        x_inp, x_mu_out = input_list[0], input_list[1]
        l1_loss = self.l1_anneal * K.sum(K.abs(x_inp - x_mu_out), axis=-1)
        return l1_loss

    def compute_output_shape(self, input_shape):
        return input_shape[0][:1]


class DistrAlignment(Layer):

    def __init__(self, latent_size, input_dims=1, **kwargs):
        self.dim_stoc = latent_size
        self.input_dims = input_dims
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.weight = self.add_weight(name='wt_weight', shape=(), initializer=lambda x: np.array(1.),
                                      trainable=False)
        super().build(input_shape)

    def calculate_loss(self, vr_prms):
        """Computes Wasserstein-distance between two variational distributions."""
        var_prms_1, var_prms_2 = vr_prms
        mu, logvar = var_prms_1[..., :self.dim_stoc], var_prms_1[..., self.dim_stoc:]
        mu_2, logvar_2 = var_prms_2[..., :self.dim_stoc], var_prms_2[..., self.dim_stoc:]
        loss = self.weight * K.sqrt(K.sum(K.square(mu - mu_2), axis=-1) +
                                    K.sum(K.square(K.sqrt(K.exp(logvar)) - K.sqrt(K.exp(logvar_2))),
                                          axis=-1))
        return loss

    def call(self, var_prms, **kwargs):
        var_prms_1 = var_prms[0]
        var_prms_2 = var_prms[1]

        return self.calculate_loss([var_prms_1, var_prms_2])
