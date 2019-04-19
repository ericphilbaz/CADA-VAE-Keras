import keras
import numpy as np
import tensorflow as tf


class CadaVaeCallback(keras.callbacks.Callback):

    def __init__(self,
                 resnet_features, attributes, labels,
                 beta_factor, beta_start, beta_end,
                 r_factor,
                 cr_factor, cr_start, cr_end,
                 alignment_factor, alignment_start, alignment_end,
                 tfcallback=None):
        super().__init__()
        self.resnet_features = resnet_features
        self.attributes = attributes
        self.labels = labels

        self.beta_factor = beta_factor
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.r_factor = r_factor
        self.cr_factor = cr_factor
        self.cr_start = cr_start
        self.cr_end = cr_end
        self.alignment_factor = alignment_factor
        self.alignment_start = alignment_start
        self.alignment_end = alignment_end

        self.tfcallback = tfcallback

    def on_epoch_begin(self, epoch, logs=None):
        """Weight Control"""

        """KL Loss"""
        if self.beta_end != 0:
            f1 = ((epoch - self.beta_start) / (self.beta_end - self.beta_start)) * self.beta_factor
            beta = min(max(f1, 0), self.beta_factor)
        else:
            beta = self.beta_factor  # no warm-up
        self.model.get_layer('kl_w').set_weights([np.array(beta)])

        """Reconstruction Loss"""
        self.model.get_layer('reconstruction').set_weights([np.array(self.r_factor)])

        """Cross Reconstruction Loss"""
        if self.cr_end != 0:
            f2 = ((epoch - self.cr_start) / (self.cr_end - self.cr_start)) * self.cr_factor
            cr_factor = min(max(f2, 0), self.cr_factor)
        else:
            cr_factor = self.cr_factor  # no warm-up
        self.model.get_layer('reconstr_cross').set_weights([np.array(cr_factor)])

        """Alignment Loss"""
        if self.alignment_end != 0:
            f3 = ((epoch - self.alignment_start) / (self.alignment_end - self.alignment_start)) * self.alignment_factor
            alignment_factor = min(max(f3, 0), self.alignment_factor)
        else:
            alignment_factor = self.alignment_factor  # no warm-up
        self.model.get_layer('alignment').set_weights([np.array(alignment_factor)])

        if self.tfcallback is not None:
            self.write_log(self.tfcallback, ['weight_beta'], self.model.get_layer('kl_w').get_weights(), epoch)
            self.write_log(self.tfcallback, ['weight_cross_reconstr'],
                           self.model.get_layer('reconstr_cross').get_weights(), epoch)
            self.write_log(self.tfcallback, ['weight_alignment'],
                           self.model.get_layer('alignment').get_weights(), epoch)
            self.write_log(self.tfcallback, ['weight_reconstruction'],
                           self.model.get_layer('reconstruction').get_weights(), epoch)

    def write_log(self, callback, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()
