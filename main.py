import argparse
import datetime
import os
import pickle
import time

import numpy as np
import tensorflow as tf
from hyperopt import STATUS_OK
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

from model.callback import CadaVaeCallback
from model.data_loader import get_data
from model.evaluation import evaluate
from model.model import build_cada_vae
from model.utils import flatten_dict

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=None)
args = parser.parse_args()

if args.gpu is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)
    print('GPU: {}'.format(args.gpu))
else:
    print('Single GPU')

# dynamically grow the memory used on the GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

data_dir = '/home/ubuntu/data/CUB/input/akata_data'

params = {
    'vae': {'lr': 1.5e-4,
            'epochs': 100,
            'batch_size': 16,
            'latent_size': 64,
            'hidden_size': {'img_enc': 1653,
                            'att_enc': 1931,
                            'img_dec': 2007,
                            'att_dec': 1996,
                            },
            'sample_like_cada_vae': True,
            },

    # Weights
    'beta': {'factor': 82.63,
             'start': 0,
             'end': 80},

    'reconstruction': {'factor': 68.43},

    'cross-reconstruction': {'factor': 422.14,
                             'start': 3,
                             'end': 75},

    'alignment': {'factor': 520.94,
                  'start': 2,
                  'end': 24},

    # Evaluation
    'samples_per_class': {'seen': 200,
                          'unseen': 400},
    'cls': {'lr': 1e-3,
            'batch_size': 32,
            'epochs': {'CUB': 22,
                       'SUN': 15,
                       'AWA1': 36,
                       'AWA2': 34,
                       }
            },

    'config': {
        'model_name': 'CadaVAE',
        'split': 'validation_akata',
        'space_hint': '_',
        'datasets': ['CUB', 'SUN', 'AWA1', 'AWA2'],
        'num_shots': 0,
    },

}

output_dir = '/output/'
output_dir = os.getcwd() + output_dir
tb_dir = None
tfcallback = None
data = {}


def model(params):
    start = time.time()

    accuracies_normal = {}
    accuracies_max = {}
    max_h_accs = []
    early_stopping = {}

    print('\n## PARAMS:')
    for key, value in flatten_dict(params).items():
        print('{}: {}'.format(key, value))
    print()

    for dataset in params['config']['datasets']:
        print('Dataset: {}'.format(dataset))
        resnet_features, attributes, labels = get_data(dataset=dataset, split=params['config']['split'],
                                                       data_path=data_dir)

        """
        Model
        """
        model, (eval_q_mu_img, eval_q_mu_att), (eval_z_img, eval_z_att) = build_cada_vae(
            latent_size=params['vae']['latent_size'],
            hidden_size_enc_img=params['vae']['hidden_size']['img_enc'],
            hidden_size_enc_att=params['vae']['hidden_size']['att_enc'],
            hidden_size_dec_img=params['vae']['hidden_size']['att_dec'],
            hidden_size_dec_att=params['vae']['hidden_size']['img_dec'],

            img_shape=resnet_features['train_seen'].shape[1:],
            semantic_shape=attributes['train_seen'].shape[1:],

            lr=params['vae']['lr'],
            sample_like_cada_vae=params['vae']['sample_like_cada_vae'],
        )

        cb = CadaVaeCallback(resnet_features=resnet_features, attributes=attributes, labels=labels,

                             beta_factor=params['beta']['factor'],
                             beta_start=params['beta']['start'],
                             beta_end=params['beta']['end'],

                             r_factor=params['reconstruction']['factor'],

                             cr_factor=params['cross-reconstruction']['factor'],
                             cr_start=params['cross-reconstruction']['start'],
                             cr_end=params['cross-reconstruction']['end'],

                             alignment_factor=params['alignment']['factor'],
                             alignment_start=params['alignment']['start'],
                             alignment_end=params['alignment']['end'],

                             tfcallback=tfcallback)

        print('## Training model')
        model.fit(x=[resnet_features['train_seen'], attributes['train_seen'],
                     resnet_features['train_seen'], attributes['train_seen']],
                  y=[resnet_features['train_seen']],
                  callbacks=[cb],
                  epochs=params['vae']['epochs'],
                  verbose=2,
                  batch_size=params['vae']['batch_size'],
                  )

        """
        Evaluate
        """

        print('## Start evaluation:')
        (s, u, h), (max_s, max_u, max_h, max_epoch), classifier = evaluate(resnet_features=resnet_features,
                                                                           attributes=attributes,
                                                                           labels=labels,
                                                                           eval_z_img=eval_z_img,
                                                                           eval_z_att=eval_z_att,
                                                                           eval_q_mu_img=eval_q_mu_img,
                                                                           samples_per_class_seen=
                                                                           params['samples_per_class']['seen'],
                                                                           samples_per_class_unseen=
                                                                           params['samples_per_class']['unseen'],
                                                                           cls_train_steps=params['cls']['epochs'][
                                                                               dataset],
                                                                           cls_batch_size=params['cls']['batch_size'],
                                                                           lr_cls=params['cls']['lr'],
                                                                           latent_size=params['vae']['latent_size'],
                                                                           num_shots=params['config']['num_shots'],
                                                                           verbose=False,
                                                                           tb_dir=tb_dir,
                                                                           )

        accuracies_normal[dataset] = (s, u, h)
        accuracies_max[dataset] = (max_s, max_u, max_h, max_epoch)
        print('Accuracies normal: {}'.format(accuracies_normal[dataset]))
        print('Accuracies max: {}:'.format(accuracies_max[dataset]))
        max_h_accs.append(max_h)
        early_stopping[dataset] = max_epoch

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        model.save_weights(output_dir + '{}_vae_rm_{}_{:.2f}.pkl'.format(dataset, params['config']['split'], max_h))
        model.save_weights(
            output_dir + '{}_vae_rm_softmax_{}_{:.2f}.pkl'.format(dataset, params['config']['split'], max_h))
        print('Saved weights')

        K.clear_session()

    end = time.time()
    print('Minutes for past run: {:.1f}'.format((end - start) / 60))

    data['loss'] = -np.mean(max_h_accs)
    data['accuracies_normal'] = accuracies_normal
    data['accuracies_max'] = accuracies_max
    data['params'] = params
    data['early_stopping_epoch_softmax'] = early_stopping
    data['time_for_one_run_in_s'] = end - start
    now = datetime.datetime.now()
    data['date'] = '{}_{}_{}_{}_{}'.format(now.year, now.month, now.day, now.hour, now.minute)

    with open(output_dir + 'results.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return {'loss': -np.mean(max_h_accs), 'status': STATUS_OK,
            'output_dir': output_dir}  # for hyperopt optimization library


if __name__ == '__main__':
    model(params)
