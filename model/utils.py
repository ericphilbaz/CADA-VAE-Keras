import numpy as np
from keras import backend as K
from keras.layers import Layer
import collections


class SampleGaussian(Layer):
    def __init__(self, dim_stochastic, avae_variant=False, **kwargs):
        self.dim_stoc = dim_stochastic
        self.avae_variant = avae_variant
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inp):
        inp_shp = K.int_shape(inp)
        if inp_shp[-1] != self.dim_stoc * 2:
            raise ValueError('Expecting stoc dim to have size: ' + str(2 * self.dim_stoc))

        mu, logvar = inp[..., :self.dim_stoc], inp[..., self.dim_stoc:]

        if not self.avae_variant:
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.0)
        else:
            # sample epsilon once and use for all dimensions
            epsilon = K.random_normal(shape=(1,), mean=0., stddev=1.0)

        z_s = mu + K.exp(logvar) * epsilon  # original code forgot the 0.5, keep mistake during reproduction

        return z_s

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.dim_stoc,)

    def get_config(self):
        config = {'dim_stoc': self.dim_stoc}
        base_config = super(SampleGaussian, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def eval_gzsl(classifier, test_X, test_Y, target_classes):
    predicted_labels = np.argmax(classifier.predict([test_X]), axis=1)
    test_labels = np.argmax(test_Y, axis=1)

    seen_classes, unseen_classes = target_classes
    seen_acc = compute_per_class_accuracy(predicted_labels, test_labels, seen_classes)
    unseen_acc = compute_per_class_accuracy(predicted_labels, test_labels, unseen_classes)

    h_acc = get_harmonic_mean(seen_acc, unseen_acc)

    return seen_acc, unseen_acc, h_acc


def compute_per_class_accuracy(predicted_labels, test_labels, target_classes):
    tmp = []
    for l in np.unique(target_classes):
        idx = np.where(l == test_labels)
        tmp.append(np.mean(test_labels[idx] == predicted_labels[idx]))
    return np.mean(tmp) * 100


def get_harmonic_mean(acc_tr, acc_ts):
    return (2 * acc_tr * acc_ts) / (acc_tr + acc_ts)


def sample_train_data_on_sample_per_class_basis(features, labels, samples_per_class):
    features_stacked = []
    labels_stacked = []

    for l in np.unique(labels):
        idx = np.where(l == labels)
        features_of_that_class = features[idx].shape[0]

        multiplier = np.ceil(max(1, samples_per_class / features_of_that_class))
        features_of_that_class = np.repeat(features[idx], multiplier, axis=0)
        labels_of_that_class = np.repeat(labels[idx], multiplier, axis=0)

        idx = np.random.choice(np.array(range(features_of_that_class.shape[0])), samples_per_class, replace=False)
        features_stacked.append(features_of_that_class[idx])
        labels_stacked.append(labels_of_that_class[idx])

    features_stacked = np.vstack(features_stacked)
    labels_stacked = np.hstack(labels_stacked)
    return features_stacked, labels_stacked


def transfer_image_features_into_training_set(resnet_features, labels):
    tmp = {'train_feat': [], 'test_feat': [], 'train_labels': [], 'test_labels': []}
    for l in np.unique(labels['test_unseen']):
        idx = np.where(l == labels['test_unseen'])
        cut_off = 5  # 5-shot
        tmp['train_feat'].append(resnet_features['test_unseen'][idx][:cut_off])
        tmp['test_feat'].append(resnet_features['test_unseen'][idx][cut_off:])
        tmp['train_labels'].append(labels['test_unseen'][idx][:cut_off])
        tmp['test_labels'].append(labels['test_unseen'][idx][cut_off:])

    resnet_features['train_unseen'] = np.vstack(tmp['train_feat'])
    resnet_features['test_unseen'] = np.vstack(tmp['test_feat'])
    labels['train_unseen_resnet'] = np.hstack(tmp['train_labels'])
    labels['test_unseen_resnet'] = np.hstack(tmp['test_labels'])

    return resnet_features, labels


def flatten_dict(d, parent_key='', sep='_'):
    """Flattens nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
