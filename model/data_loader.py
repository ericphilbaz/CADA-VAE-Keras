import copy
import pickle
import sys

import numpy as np
import scipy.io as sio
from sklearn import preprocessing


def map_label(label, classes):
    mapped_label = np.empty(shape=label.shape, dtype=int)
    for i in range(classes.shape[0]):
        mapped_label[label == classes[i]] = i

    return mapped_label


class Dataloader(object):
    def __init__(self, dataset, aux_datasource, data_path, split='validation'):
        """
        Split, exemplary for CUB:

        'test': 150 seen classes, 50 unseen classes
        'validation':
            - 100 val-seen classes, 50 val-unseen classes (zero-shot validation split), _all_ available data (following
            author of cada-vae paper)
            - 100 val-seen classes are split into val-train and val-test
        """

        sys.path.append(data_path)

        self.data_path = data_path
        self.split = split
        self.dataset = dataset
        self.auxiliary_data_source = aux_datasource

        self.all_data_sources = ['resnet_features'] + [self.auxiliary_data_source]

        if self.dataset == 'CUB':
            self.datadir = self.data_path + '/CUB/'
        elif self.dataset == 'SUN':
            self.datadir = self.data_path + '/SUN/'
        elif self.dataset == 'AWA1':
            self.datadir = self.data_path + '/AWA1/'
        elif self.dataset == 'AWA2':
            self.datadir = self.data_path + '/AWA2/'

        self.read_matdataset()
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self):

        path = self.datadir + 'res101.mat'
        matcontent = sio.loadmat(path)
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1

        path = self.datadir + 'att_splits.mat'
        matcontent = sio.loadmat(path)  # numpy array index starts from 0, matlab starts from 1

        if self.split == 'validation':
            # ZS-validation split
            train_loc = matcontent['train_loc'].squeeze() - 1  # seen
            val_unseen_loc = matcontent['val_loc'].squeeze() - 1  # unseen

            # split seen data into val-train and val-test
            tmp_train_seen = []
            tmp_test_seen = []
            for l in np.unique(label[train_loc]):
                idx = np.where(l == label[train_loc])
                cut_off = int(train_loc[idx].shape[0] * 0.80)  # cut-off point mimicking train/test-ratio of real data
                # exact ratios:
                #   - CUB: 0.7997
                #   - SUN: 0.7999999999999998
                #   - AWA1: 0.7994623145114395
                #   - AWA2: 0.7988800443020644
                tmp_train_seen.append(train_loc[idx][:cut_off])
                tmp_test_seen.append(train_loc[idx][cut_off:])

            trainval_loc = np.hstack(tmp_train_seen)  # train_seen (from ZS-validation split)
            test_seen_loc = np.hstack(tmp_test_seen)  # test_seen (from ZS-validation split)
            test_unseen_loc = val_unseen_loc  # train_unseen (from  ZS-validation split)

        elif self.split == 'test':
            trainval_loc = matcontent['trainval_loc'].squeeze() - 1  # train_seen (GZSL)
            test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1  # test_seen (GZSL)
            test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1  # test_unseen (GZSL)

            # compute ration train_seen/(train_seen+test_seen)
            # tmp = []
            # for l in np.unique(label[test_seen_loc]):
            #     train = np.where(label[trainval_loc] == l)[0].shape[0]
            #     test = np.where(label[test_seen_loc] == l)[0].shape[0]
            #     tmp.append(train / (train + test))
            # print(np.mean(tmp))

        if self.auxiliary_data_source == 'attributes':
            self.aux_data = matcontent['att'].T
        else:
            if self.dataset != 'CUB':
                print('the specified auxiliary datasource is not available for this dataset')
            else:

                with open(self.datadir + 'CUB_supporting_data.p', 'rb') as h:
                    x = pickle.load(h)
                    self.aux_data = x[self.auxiliary_data_source]

                print('loaded ', self.auxiliary_data_source)

        """Verify correct dataloading for validation"""
        if self.split == 'ValidationAkata':
            if self.dataset == 'CUB':
                num_train_val_classes = 100
            elif self.dataset == 'SUN':
                num_train_val_classes = 580
            elif self.dataset == 'AWA1':
                num_train_val_classes = 27
            elif self.dataset == 'AWA2':
                num_train_val_classes = 27

            if self.split == 'validation':
                if np.unique(label[trainval_loc]).shape[0] == num_train_val_classes:
                    print('Loaded validation data.')
                else:
                    raise Exception('Incorrect validation data loaded.')
            else:
                print('Loaded test data.')

        # normalize resnet features to be between zero and one
        #  -    attribute data is between 0 and 0.29
        #  -    sentence data is between ~0.17 and 0.18
        scaler = preprocessing.MinMaxScaler()
        train_feature = scaler.fit_transform(feature[trainval_loc])
        test_seen_feature = scaler.fit_transform(feature[test_seen_loc])
        test_unseen_feature = scaler.fit_transform(feature[test_unseen_loc])

        train_feature = train_feature
        test_seen_feature = test_seen_feature
        test_unseen_feature = test_unseen_feature

        train_label = label[trainval_loc]
        test_unseen_label = label[test_unseen_loc]
        test_seen_label = label[test_seen_loc]

        self.seenclasses = np.unique(train_label)
        self.novelclasses = np.unique(test_unseen_label)
        self.ntrain = train_feature.shape[0]
        self.ntrain_class = self.seenclasses.shape[0]
        self.ntest_class = self.novelclasses.shape[0]
        self.train_class = copy.deepcopy(self.seenclasses)
        self.allclasses = np.arange(0, self.ntrain_class + self.ntest_class)

        self.train_mapped_label = map_label(train_label, self.seenclasses)

        self.data = {'train_seen': {}}
        self.data['train_seen']['resnet_features'] = train_feature
        self.data['train_seen']['labels'] = train_label
        self.data['train_seen'][self.auxiliary_data_source] = self.aux_data[train_label]

        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen'][self.auxiliary_data_source] = self.aux_data[test_seen_label]  # only for analysis
        self.data['test_seen']['resnet_features'] = test_seen_feature
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = test_unseen_feature
        self.data['test_unseen'][self.auxiliary_data_source] = self.aux_data[test_unseen_label]
        self.data['test_unseen']['labels'] = test_unseen_label

        self.novelclass_aux_data = self.aux_data[self.novelclasses]
        self.seenclass_aux_data = self.aux_data[self.seenclasses]


def get_validation_data(resnet_features, attributes, labels):
    """
    Exemplary for CUB:
    1. Splits classes into 100 seen and 50 unseen classes (todo: use classes indicated by official split)
    2. Split seen classes into train_seen and test_seen
    """
    trainval_classes = np.random.choice(np.unique(labels['train_seen']), 100, replace=False)
    val_classes = np.array(list(set(labels['train_seen']) - set(trainval_classes)))

    """Seen Classes"""
    resnet_features_val = {}
    attributes_val = {}
    labels_val = {}

    resnet_features_val['train_seen'] = []
    resnet_features_val['test_seen'] = []
    attributes_val['train_seen'] = []
    labels_val['train_seen'] = []
    labels_val['test_seen'] = []

    for l in trainval_classes:
        idx = np.where(l == labels['train_seen'])
        cut_off = int(resnet_features['train_seen'][idx].shape[0] * 0.80)

        resnet_features_val['train_seen'].append(resnet_features['train_seen'][idx][:cut_off])
        resnet_features_val['test_seen'].append(resnet_features['train_seen'][idx][cut_off:])
        attributes_val['train_seen'].append(attributes['train_seen'][idx][:cut_off])

        labels_val['train_seen'].append(labels['train_seen'][idx][:cut_off])
        labels_val['test_seen'].append(labels['train_seen'][idx][cut_off:])

    resnet_features_val['train_seen'] = np.vstack(resnet_features_val['train_seen'])
    resnet_features_val['test_seen'] = np.vstack(resnet_features_val['test_seen'])
    attributes_val['train_seen'] = np.vstack(attributes_val['train_seen'])
    labels_val['train_seen'] = np.hstack(labels_val['train_seen'])
    labels_val['test_seen'] = np.hstack(labels_val['test_seen'])

    """Unseen Classes"""
    resnet_features_val['test_unseen'] = []
    attributes_val['test_unseen'] = []
    labels_val['test_unseen'] = []

    for l in val_classes:
        idx = np.where(l == labels['train_seen'])
        resnet_features_val['test_unseen'].append(resnet_features['train_seen'][idx])
        attributes_val['test_unseen'].append(attributes['train_seen'][idx])

        labels_val['test_unseen'].append(labels['train_seen'][idx])

    resnet_features_val['test_unseen'] = np.vstack(resnet_features_val['test_unseen'])
    attributes_val['test_unseen'] = np.vstack(attributes_val['test_unseen'])
    labels_val['test_unseen'] = np.hstack(labels_val['test_unseen'])

    return resnet_features_val, attributes_val, labels_val


def get_data(dataset, split, data_path):
    """
    :param split: either 'validation_akata', 'validation_own' or 'test'
    """
    if split == 'validation_akata':
        dataset = Dataloader(dataset=dataset, aux_datasource='attributes',
                             data_path=data_path,
                             split='validation')
    elif split == 'validation_own' or split == 'test':
        dataset = Dataloader(dataset=dataset, aux_datasource='attributes',
                             data_path=data_path,
                             split='test')

    resnet_features = {'train_seen': dataset.data['train_seen']['resnet_features'],
                       'test_seen': dataset.data['test_seen']['resnet_features'],
                       'test_unseen': dataset.data['test_unseen']['resnet_features']}

    attributes = {'train_seen': dataset.data['train_seen']['attributes'],
                  'test_seen': dataset.data['test_seen']['attributes'],
                  'test_unseen': dataset.data['test_unseen']['attributes']}

    labels = {'train_seen': dataset.data['train_seen']['labels'], 'test_seen': dataset.data['test_seen']['labels'],
              'test_unseen': dataset.data['test_unseen']['labels']}

    if split == 'validation_own':
        resnet_features, attributes, labels = get_validation_data(resnet_features, attributes, labels)

    return resnet_features, attributes, labels
