import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
try:
    import cPickle as pickle
except:
    import pickle

class Dataset:

    def __init__(self, 
                 data_path,
                 prediction_data_set = False,
                 n_context = 1,
                 stride = 1,
                 segment_length = 100,
                 batch_size = 128,
                 split_train_dev = True,
                 shuffle = True,
                 **kwargs):
        """
        feature_dir and label_dir should both be lists of pickle file, even if they only contains one file.
        both args should be aligned with each other!

        label_dir is for controller pilot identification task.
        it can be extended to transcription in the future.
        """
        self.data_path = data_path
        self.prediction_data_set = prediction_data_set
        self.n_context = n_context
        self.stride = stride
        self.segment_length = segment_length
        self.batch_size = batch_size

        self.split_train_dev = split_train_dev
        self.shuffle = shuffle

        if self.prediction_data_set:
            self.prediction_inputs, self.prediction_seq_lengths, self.prediction_files = self.process_prediction_feature()
            self.split_train_dev = False
            self.shuffle = False
            self.n_prediction_data_set = self.prediction_inputs.shape[0]
        else:
            total_inputs, total_targets, total_seq_lengths = self.process_feature_label()

        if self.split_train_dev:
            self.train_dev_ratio = kwargs.get('train_dev_ratio', 0.9)
            self.train_inputs, \
                self.dev_inputs, \
                    self.train_targets, \
                        self.dev_targets, \
                            self.train_seq_lens, \
                                self.dev_seq_lens = train_test_split(total_inputs, 
                                                                     total_targets,
                                                                     total_seq_lengths,
                                                                     train_size = self.train_dev_ratio,
                                                                     random_state = 101)
            self.n_train_data_set = self.train_inputs.shape[0]
            self.n_dev_data_set = self.dev_inputs.shape[0]

        self.idx = kwargs.get('idx', 0)

    def next_batch(self, train_dev_test = 'train'):
        if train_dev_test == 'train':
            n_sample = self.n_train_data_set
        elif train_dev_test == 'dev':
            n_sample = self.n_dev_data_set
        elif train_dev_test == 'test':
            n_sample = self.n_prediction_data_set
        else:
            raise ValueError('train_dev_test must be train or dev or test')

        if self.idx >= n_sample:
            self.idx = 0
            if self.shuffle:
                self.train_inputs, self.train_targets, self.train_seq_lens = shuffle(self.train_inputs, self.train_targets, self.train_seq_lens)

        if train_dev_test == 'train':
            endidx = min(self.idx + self.batch_size, self.n_train_data_set)
            batch_inputs = self.train_inputs[self.idx:endidx, :, :]
            batch_targets = self.train_targets[self.idx:endidx, :, :]
            batch_seq_lens = self.train_seq_lens[self.idx:endidx]
        elif train_dev_test == 'dev':
            endidx = min(self.idx + self.batch_size, self.n_dev_data_set)
            batch_inputs = self.dev_inputs[self.idx:endidx, :, :]
            batch_targets = self.dev_targets[self.idx:endidx, :, :]
            batch_seq_lens = self.dev_seq_lens[self.idx:endidx]
        elif train_dev_test == 'test':
            endidx = min(self.idx + self.batch_size, self.n_prediction_data_set)
            batch_inputs = self.prediction_inputs[self.idx:endidx, :, :]
            batch_targets = None
            batch_seq_lens = self.prediction_seq_lengths[self.idx:endidx]
        else:
            raise ValueError('train_dev_test must be train or dev or test')
        
        self.idx += self.batch_size

        return batch_inputs, batch_targets, batch_seq_lens

    def process_feature_label(self):
        train_inputs = []
        targets = []
        seq_lengths = []
        
        feature_files = []
        label_files = []

        for data_file in os.listdir(self.data_path):
            if data_file.endswith('_feature.pkl'):
                feature_files.append(data_file)
        label_files = [x.replace('_feature.pkl', '_label.pkl') for x in feature_files]

        i = 0
        for feature_file, label_file in zip(feature_files, label_files):
            feature_file = os.path.join(self.data_path, feature_file)
            label_file = os.path.join(self.data_path, label_file)
            train_input, target, seq_length = self.__process_feature_label__(feature_file,
                                                                             label_file,
                                                                             self.n_context,
                                                                             self.stride,
                                                                             self.segment_length)
            if i == 0:
                train_inputs = train_input.copy()
                targets = target.copy()
                seq_lengths = seq_length.copy()
            else:
                train_inputs = np.concatenate((train_inputs, train_input), axis = 0)
                targets = np.concatenate((targets, target), axis = 0)
                seq_lengths = np.concatenate((seq_lengths, seq_length), axis = 0)

            i += 1

        train_inputs = np.asarray(train_inputs, dtype = 'float32')
        targets = np.asarray(targets, dtype = 'int32')
        seq_lengths = np.asarray(seq_lengths, dtype = 'int32')
        print(train_inputs.shape)
        print(targets.shape)
        print(seq_lengths.shape)
        return train_inputs, targets, seq_lengths

    def process_prediction_feature(self):
        feature_inputs = []
        seq_lengths = []
        
        feature_files = []
        
        for data_file in os.listdir(self.data_path):
            if data_file.endswith('_feature.pkl'):
                feature_files.append(data_file)
        # No need for target file dir
        i = 0
        for feature_file in feature_files:
            feature_file = os.path.join(self.data_path, feature_file)
            
            train_input, _, seq_length = self.__process_feature_label__(feature_file,
                                                                        None,
                                                                        self.n_context,
                                                                        self.stride,
                                                                        self.segment_length)
            if i == 0:
                feature_inputs = train_input.copy()
                seq_lengths = seq_length.copy()
            else:
                feature_inputs = np.concatenate((feature_inputs, train_input), axis = 0)
                seq_lengths = np.concatenate((seq_lengths, seq_length), axis = 0)

            i += 1

        feature_inputs = np.asarray(feature_inputs, dtype = 'float32')
        seq_lengths = np.asarray(seq_lengths, dtype = 'int32')
        print(feature_inputs.shape)
        print(seq_lengths.shape)
        return feature_inputs, seq_lengths, feature_files


    def __process_feature_label__(self,
                                feature_file,
                                label_file,
                                n_context = 1,
                                stride = 1,
                                segment_length = 100):
        with open(feature_file, 'rb') as f:
            original_input = pickle.load(f, encoding='latin1')
            # normalize
            original_input = ((original_input - np.mean(original_input, axis = 0))/np.std(original_input, axis = 0))
            if stride == 1:
                pass
            else:
                original_input = original_input[::stride]
        # time axis should be the row
        time_dim, feature_dim = original_input.shape

        if n_context < 0:
            raise ValueError("n_context must be >= 0")
        elif n_context == 0:
            train_input = original_input
        else:
            pad_empty = np.zeros(shape = (n_context, original_input.shape[1]))
            tmp_input = np.append(pad_empty, original_input, axis = 0)
            tmp_input = np.append(tmp_input, pad_empty, axis = 0)

            train_input = np.zeros(shape = (time_dim, feature_dim * (1 + 2 * n_context)))

            for i in range(1 + 2 * n_context):
                train_input[:, i*feature_dim: (i+1)*feature_dim] = tmp_input[i: (i+time_dim), :]

        
        if segment_length is not None:
            pad_len = segment_length - time_dim % segment_length

            seq_length = [segment_length] * (time_dim // segment_length)
            seq_length.append(segment_length - pad_len)
            seq_length = np.asarray(seq_length)

            train_input = np.append(train_input, np.zeros(shape = (pad_len, train_input.shape[1])), axis = 0)
            # print(train_input.shape)
            train_input = train_input.reshape((-1, segment_length, train_input.shape[1]))

        else:
            raise NotImplementedError("Not implemented!")

        ######################################################
        #######                TARGET                  #######
        ######################################################

        if label_file is None:
            target = None
        else:
            with open(label_file, 'rb') as fl:
                target_flat = pickle.load(fl, encoding='latin1')
            # one hot
            # target_flat is a vector of 0, 1, 2 where 0 is vacant, 1 is pilot and 2 is controller.
            target = np.zeros(shape = (target_flat.size, target_flat.max() + 1))
            target[np.arange(target_flat.size), target_flat] = 1

            _, target_dim = target.shape

            if segment_length is not None:
                pad_len = segment_length - time_dim % segment_length
                target = np.append(target, np.zeros(shape = (pad_len, target_dim)), axis = 0)
                target = target.reshape((-1, segment_length, target_dim))
            else:
                raise NotImplementedError("Not implemented!")

        return train_input, target, seq_length