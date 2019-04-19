import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model

from model.utils import eval_gzsl


def build_linear_softmax(latent_size, lr_cls=1e-3, n_classes=150):
    inp = Input(shape=(latent_size,))
    out = Dense(n_classes, activation='softmax')(inp)

    model = Model(inputs=[inp], outputs=[out])

    opt = keras.optimizers.adam(lr=lr_cls, beta_1=0.5, beta_2=0.999, epsilon=1e-07)
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    return model


class SoftmaxCB(keras.callbacks.Callback):

    def __init__(self,
                 emb_vae, labels,
                 tfcallback=None, validate=True, verbose=True):
        super().__init__()
        self.emb_vae = emb_vae
        self.labels = labels
        self.tfcallback = tfcallback
        self.validate = validate
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        """Validation"""
        if epoch == 0:
            self.model.history.history.update({'s_acc': []})
            self.model.history.history.update({'u_acc': []})
            self.model.history.history.update({'h_acc': []})

        if self.validate:
            seen_acc, unseen_acc, h_acc = eval_gzsl(classifier=self.model,
                                                    test_X=self.emb_vae['test_all'],
                                                    test_Y=self.labels['test_all'],
                                                    target_classes=[self.labels['test_seen'],
                                                                    self.labels['test_unseen']])
            self.model.history.history['s_acc'] += [seen_acc]
            self.model.history.history['u_acc'] += [unseen_acc]
            self.model.history.history['h_acc'] += [h_acc]
            if self.verbose:
                print('S: {:.1f}; U: {:.1f}; H: {:.1f} (max_H: {:.1f} at epoch {}))'.format(
                    seen_acc,
                    unseen_acc,
                    h_acc,
                    np.max(self.model.history.history['h_acc']),
                    np.argmax(self.model.history.history['h_acc']) + 1))

            if self.tfcallback is not None:
                self.write_log(self.tfcallback, ['softmax_s_gzsl'], [seen_acc], epoch)
                self.write_log(self.tfcallback, ['softmax_u_'], [unseen_acc], epoch)
                self.write_log(self.tfcallback, ['softmax_h_gzsl'], [h_acc], epoch)

    def write_log(self, callback, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()
