import keras
import numpy as np

from model.softmax import build_linear_softmax, SoftmaxCB
from model.utils import sample_train_data_on_sample_per_class_basis, transfer_image_features_into_training_set, \
    eval_gzsl


def evaluate(resnet_features, attributes, labels,
             eval_z_img, eval_z_att, eval_q_mu_img,
             samples_per_class_seen,
             samples_per_class_unseen,
             cls_train_steps,
             cls_batch_size,
             lr_cls,
             latent_size,
             num_shots=0,
             verbose=False,
             tb_dir=None):
    """Evaluation Generalized Zero-Shot.

    num_shots>0 implementation not yet finished.
    """

    """
    Normalize labels, i.e. [2,5,7] -> [0,1,2]
    """
    # train_seen
    tmp = np.ones(labels['train_seen'].shape) * -1
    i = 0
    for l in np.unique(labels['train_seen']):
        idx = np.where(l == labels['train_seen'])
        tmp[idx] = i
        i += 1
    labels['train_seen'] = tmp

    # test_seen
    tmp = np.ones(labels['test_seen'].shape) * -1
    i = 0
    for l in np.unique(labels['test_seen']):
        idx = np.where(l == labels['test_seen'])
        tmp[idx] = i
        i += 1
    labels['test_seen'] = tmp

    # test_unseen
    tmp = np.ones(labels['test_unseen'].shape) * -1
    for l in np.unique(labels['test_unseen']):
        idx = np.where(l == labels['test_unseen'])
        tmp[idx] = i
        i += 1
    labels['test_unseen'] = tmp

    """
    Sample embeddings
    """
    if num_shots > 0:
        resnet_features, labels = transfer_image_features_into_training_set(resnet_features, labels)

    number_of_classes = np.unique(labels['test_seen']).shape[0] + np.unique(labels['test_unseen']).shape[0]

    resnet_features['train_seen_stacked'], \
    labels['train_seen_stacked'] = sample_train_data_on_sample_per_class_basis(
        resnet_features['train_seen'],
        labels['train_seen'],
        samples_per_class_seen)

    if num_shots == 0:
        # sample unseen embeddings from attributes
        attributes['test_unseen_stacked'], \
        labels['test_unseen_stacked'] = sample_train_data_on_sample_per_class_basis(
            attributes['test_unseen'],
            labels['test_unseen'],
            samples_per_class_unseen)

    elif num_shots > 0:
        # sample unseen embeddings from resnet features
        resnet_features['train_unseen_stacked'], \
        labels['train_unseen_stacked'] = sample_train_data_on_sample_per_class_basis(
            resnet_features['train_unseen'],
            labels['train_unseen_resnet'],
            int(samples_per_class_unseen))

    """
    Get variational parameter
    """
    emb_vae = {'train_seen': eval_z_img.predict([resnet_features['train_seen_stacked']])}

    if num_shots == 0:
        # unseen varparams from attributes
        emb_vae['train_unseen'] = eval_z_att.predict([attributes['test_unseen_stacked']])
        labels['train_all'] = keras.utils.to_categorical(
            np.hstack([labels['train_seen_stacked'], labels['test_unseen_stacked']]),
            num_classes=number_of_classes)

    elif num_shots > 0:
        # unseen varparams from resnet features
        emb_vae['train_unseen'] = eval_z_img.predict([resnet_features['train_unseen_stacked']])
        labels['train_all'] = keras.utils.to_categorical(
            np.hstack([labels['train_seen_stacked'], labels['train_unseen_stacked']]),
            num_classes=number_of_classes)

    emb_vae['train_all'] = np.vstack([emb_vae['train_seen'], emb_vae['train_unseen']])

    emb_vae['test_seen'] = eval_q_mu_img.predict([resnet_features['test_seen']])
    emb_vae['test_unseen'] = eval_q_mu_img.predict([resnet_features['test_unseen']])
    emb_vae['test_all'] = np.vstack([emb_vae['test_seen'], emb_vae['test_unseen']])

    if num_shots == 0:
        # unseen test samples are _all_ unseen image samples
        labels['test_all'] = keras.utils.to_categorical(np.hstack([labels['test_seen'], labels['test_unseen']]),
                                                        num_classes=number_of_classes)

    elif num_shots > 0:
        # unseen test samples are unseen image samples that are not in training set
        labels['test_all'] = keras.utils.to_categorical(np.hstack([labels['test_seen'], labels['test_unseen_resnet']]),
                                                        num_classes=number_of_classes)

    """
    Softmax
    """
    if tb_dir is not None:
        tfcb = keras.callbacks.TensorBoard(log_dir=tb_dir + '_softmax')
        cb_list = [tfcb]
    else:
        tfcb = None
        cb_list = []

    classifier = build_linear_softmax(lr_cls=lr_cls,
                                      latent_size=latent_size,
                                      n_classes=number_of_classes)

    cb = SoftmaxCB(emb_vae=emb_vae, labels=labels,
                   tfcallback=tfcb, validate=True, verbose=verbose)

    cb_list += [cb]

    history = classifier.fit(emb_vae['train_all'], labels['train_all'],
                             epochs=cls_train_steps,
                             verbose=0,
                             batch_size=cls_batch_size,
                             callbacks=cb_list)

    s, u, h = eval_gzsl(classifier,
                        test_X=emb_vae['test_all'],
                        test_Y=labels['test_all'],
                        target_classes=[labels['test_seen'], labels['test_unseen']])

    max_h = np.max(history.history['h_acc'])
    max_idx = np.argmax(history.history['h_acc'])
    max_epoch = max_idx + 1
    max_s = history.history['s_acc'][max_idx]
    max_u = history.history['u_acc'][max_idx]

    return (s, u, h), (max_s, max_u, max_h, max_epoch), classifier
