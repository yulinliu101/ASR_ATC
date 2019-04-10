import tensorflow as tf
import numpy as np
import os
from configparser import ConfigParser
from rnn import BiRNN

from dataset import Dataset
import logging
import time

from tensorflow.python.client import device_lib
from sklearn.metrics import confusion_matrix

import pickle


def get_available_gpus():
    """
    Returns the number of GPUs available on this system.
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def check_if_gpu_available(gpu_name):
    """
    Returns boolean of if a specific gpu_name (string) is available
    On the system
    """
    list_of_gpus = get_available_gpus()
    if gpu_name not in list_of_gpus:
        return False
    else:
        return True

class trainRNN:
    def __init__(self, 
                 conf_path,
                 predict = False,
                 model_name=None
                 ):

        self.conf_path = conf_path
        self.predict = predict
        self.model_name = model_name
        self.load_configs()
        # if don't have gpu, then set device to be cpu
        if not check_if_gpu_available(self.tf_device):
            self.tf_device = '/cpu:0'

        logging.info('Using device %s for main computations', self.tf_device)
        
        self.set_up_directories(self.model_name)

        if not self.predict:

            self.dataset = Dataset(data_path = self.data_dir,
                                   n_context = self.n_context,
                                   stride = 1,
                                   segment_length = self.segment_length,
                                   batch_size = self.batch_size,
                                   split_train_dev = True,
                                   shuffle = self.shuffle_data_after_epoch)
            print('dev set has %.2f zeros.'%np.mean(self.dataset.dev_targets[:, :, 0] == 1))
            print('dev set has %.2f ones.'%np.mean(self.dataset.dev_targets[:, :, 1] == 1))
            print('dev set has %.2f twos.'%np.mean(self.dataset.dev_targets[:, :, 2] == 1))

    def load_configs(self):
        parser = ConfigParser(os.environ)
        if not os.path.exists(self.conf_path):
            raise IOError("Configuration file '%s' does not exist" % self.conf_path)
        logging.info('Loading config from %s', self.conf_path)
        parser.read(self.conf_path)

        # set which set of configs to import
        config_header = 'nn'

        logger.info('config header: %s', config_header)

        self.epochs = parser.getint(config_header, 'epochs')
        logger.debug('self.epochs = %d', self.epochs)

        self.network_type = parser.get(config_header, 'network_type')

        # number of feature length
        self.n_input = parser.getint(config_header, 'n_input')
        # number of target length
        self.n_character = parser.getint(config_header, 'n_character')
        # Number of contextual samples to include
        self.n_context = parser.getint(config_header, 'n_context')
        self.segment_length = parser.getint(config_header, 'segment_length')

        self.batch_size = parser.getint(config_header, 'batch_size')
        self.model_dir = parser.get(config_header, 'model_dir')

        self.data_dir = parser.get(config_header, 'data_dir')

        # set the session name
        self.session_name = '{}_{}'.format(self.network_type, time.strftime("%Y%m%d-%H%M%S"))
        sess_prefix_str = 'develop'
        if len(sess_prefix_str) > 0:
            self.session_name = '{}_{}'.format(sess_prefix_str, self.session_name)

        # How often to save the model
        self.SAVE_MODEL_EPOCH_NUM = parser.getint(config_header, 'SAVE_MODEL_EPOCH_NUM')

        # decode dev set after N epochs
        self.VALIDATION_EPOCH_NUM = parser.getint(config_header, 'VALIDATION_EPOCH_NUM')

        # determine if the data input order should be shuffled after every epic
        self.shuffle_data_after_epoch = parser.getboolean(config_header, 'shuffle_data_after_epoch')

        # set up GPU if available
        self.tf_device = str(parser.get(config_header, 'tf_device'))

        
        # optimizer
        self.beta1 = parser.getfloat(config_header, 'beta1')
        self.beta2 = parser.getfloat(config_header, 'beta2')
        self.epsilon = parser.getfloat(config_header, 'epsilon')
        self.learning_rate = parser.getfloat(config_header, 'learning_rate')
        logger.debug('self.learning_rate = %.6f', self.learning_rate)

    def set_up_directories(self, model_name):
        # Set up model directory
        self.model_dir = os.path.join(os.getcwd(), self.model_dir)
        # summary will contain logs
        self.SUMMARY_DIR = os.path.join(
            self.model_dir, "summary", self.session_name)
        # session will contain models
        self.SESSION_DIR = os.path.join(
            self.model_dir, "session", self.session_name)

        if not self.predict:
            if not os.path.exists(self.SESSION_DIR):
                os.makedirs(self.SESSION_DIR)
            if not os.path.exists(self.SUMMARY_DIR):
                os.makedirs(self.SUMMARY_DIR)

        # set the model name and restore if not None
        if model_name is not None:
            tmpSess = os.path.join(self.model_dir, "session")
            self.restored_model_path = os.path.join(tmpSess, model_name)
        else:
            self.restored_model_path = None

    def run_model(self, 
                  train_from_model = False,
                  prediction_data_dir = None):
                # define a graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            # with tf.device(self.tf_device):
            self.launchGraph()

            self.sess = tf.Session()
            # self.sess.run(tf.local_variables_initializer())
                # initialize the summary writer
            self.writer = tf.summary.FileWriter(self.SUMMARY_DIR, graph=self.sess.graph)

            # Add ops to save and restore all the variables
            self.saver = tf.train.Saver()

            section = '\n{0:=^40}\n'

            if self.restored_model_path is not None:
                self.saver.restore(self.sess, self.restored_model_path)
                if train_from_model is True:
                    logger.info(section.format('Run training epoch from restored model %s'%self.restored_model_path))
                    self.run_training_epoch()
                else:
                    logger.info(section.format('Restore model from %s'%self.restored_model_path))
                    if self.predict:
                        logger.info("=============== Start predicting ... ==============")
                        predictions = self.run_prediction(prediction_data_dir)
                        with open(os.path.join(prediction_data_dir, 'prediction.pkl'), 'wb') as wf:
                            pickle.dump(predictions, wf, protocol = 2)
                        return predictions
                    else:
                        pass
            else:
                self.sess.run(tf.global_variables_initializer())
                logger.info("Start Training ...")
                self.run_training_epoch()

            # save train summaries to disk
            self.writer.flush()

            self.sess.close()

    def launchGraph(self):
        # define placeholder
        featureLength = self.n_input + 2 * self.n_context * self.n_input
        targetLength = self.n_character
        self.input_tensor = tf.placeholder(dtype = tf.float32, shape = [None, None, featureLength], name = 'feature_map')
        # self.target = tf.sparse_placeholder(dtype = tf.int32, name = 'target')
        self.target = tf.placeholder(dtype = tf.int32, shape = [None, None, targetLength], name = 'traget')
        self.seq_length = tf.placeholder(dtype = tf.int32, shape = [None], name = 'seq_length')

        # set up network
        if self.network_type == 'simple_lstm':
            raise NotImplementedError("not finished")
            # logits, summary_op = SimpleLSTM(self.conf_path, input_tensor, seq_length)
        elif self.network_type == 'BiRNN':
            self.logits, summary_op = BiRNN(self.conf_path, self.input_tensor, self.seq_length, self.n_input, self.n_context)
        elif self.network_type == 'BiRNN_FC':
            raise NotImplementedError("not finished")
            # logits, summary_op = BiRNN_FC(self.conf_path, input_tensor, seq_length, self.n_input, self.n_context)
        else:
            raise ValueError("network type not implemented")

        self.summary_op = tf.summary.merge([summary_op])

        # define loss
        reshape_labels = tf.reshape(self.target, shape = (-1, targetLength)) # has the shape of (batch_size * seg_len, targetLength)
        reshape_logits = tf.reshape(self.logits, shape = (-1, targetLength))
        with tf.name_scope('loss'):
            
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = reshape_labels, 
                                                                   logits = reshape_logits, 
                                                                   dim = -1)
            # self.loss = tf.nn.ctc_loss(self.target, self.logits, self.seq_length, ctc_merge_repeated = False, time_major = True)
            self.avg_loss = tf.reduce_mean(self.loss)
            # self.loss_summary = tf.summary.scalar("avg_loss", self.avg_loss)

            with tf.device('/cpu:0'):
	            self.loss_placeholder = tf.placeholder(dtype = tf.float32, shape = [])
	            self.loss_summary = tf.summary.scalar("training_avg_loss", self.loss_placeholder)
            

        # setup optimizer
        with tf.name_scope('training_optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                   beta1=self.beta1,
                                                   beta2=self.beta2,
                                                   epsilon=self.epsilon).minimize(self.avg_loss)

        # "Fake" decoder
        with tf.name_scope('decoder'):
            self.label_pred = tf.argmax(reshape_logits, axis = 1)

        # define accuracy metrics and summary of statistics
        with tf.name_scope('accuracy'), tf.device('/cpu:0'):
            equality = tf.equal(tf.argmax(reshape_labels, axis = 1), self.label_pred)
            self.accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
            # self.accuracy, _ = tf.metrics.accuracy(labels = tf.argmax(reshape_labels, axis = 0), 
                                                   # predictions = self.label_pred)

            self.accuracy_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
            self.train_accuracy_op = tf.summary.scalar(
                "train_label_accuracy", self.accuracy_placeholder)
            self.dev_accuracy_op = tf.summary.scalar(
                "validation_label_accuracy", self.accuracy_placeholder)
        return 

    def run_training_epoch(self):
        train_start_time = time.time()
        for epoch in range(self.epochs):
            is_checkpoint_step, is_validation_step = self.validation_and_checkpoint_check(epoch)
            epoch_start_time = time.time()
            train_epoch_loss, epoch_accuracy = self.run_batches(self.dataset,
                                                                epoch,
                                                                'train')

            epoch_elap_time = time.time() - epoch_start_time

            log = 'Epoch {}/{}, train_cost: {:.3f}, \
                   training_labeling_error: {:.3f}, \
                   elapsed_time: {:.2f} sec \n'
            logger.info(log.format(
                epoch + 1,
                self.epochs,
                train_epoch_loss,
                epoch_accuracy,
                epoch_elap_time))

            summary_line = self.sess.run(self.train_accuracy_op, 
                                         feed_dict = {self.accuracy_placeholder: epoch_accuracy})
            self.writer.add_summary(summary_line, epoch)

            summary_line = self.sess.run(self.loss_summary, 
                                         feed_dict = {self.loss_placeholder: train_epoch_loss})
            self.writer.add_summary(summary_line, epoch)

            if (epoch + 1 == self.epochs) or is_validation_step:
                logger.info('==============================')
                logger.info('Validating ...')
                dev_accuracy = self.run_dev_epoch(epoch)
                logger.info('==============================')

            if (epoch + 1 == self.epochs) or is_checkpoint_step:
                save_path = self.saver.save(self.sess, os.path.join(self.SESSION_DIR, 'model.ckpt'), epoch)
                logger.info("Model saved to {}".format(save_path))
        train_elap_time = time.time() - train_start_time
        logger.info('Training complete, total duration: {:.2f} min'.format(train_elap_time / 60))

        return

    def run_dev_epoch(self, epoch):
        _, dev_accuracy = self.run_batches(self.dataset,
                                           epoch,
                                           'dev')
        logger.info('Prediction accuracy on dev set: {:.3f}'.format(dev_accuracy))

        summary_line = self.sess.run(self.dev_accuracy_op, feed_dict = {self.accuracy_placeholder: dev_accuracy})
        self.writer.add_summary(summary_line, epoch)
        return dev_accuracy

    def run_prediction(self, 
                       prediction_data_dir):
        prediction_dataset = Dataset(data_path = prediction_data_dir,
                                     prediction_data_set = True,
                                     n_context = self.n_context,
                                     stride = 1,
                                     segment_length = self.segment_length,
                                     batch_size = self.batch_size,
                                     split_train_dev = False,
                                     shuffle = False)
        print(prediction_dataset.prediction_files)
        pred_label = self.run_batches(prediction_dataset,
                                      epoch = 0,
                                      train_dev_test = 'test')
        return pred_label

    def validation_and_checkpoint_check(self,
                                        epoch):
        # initially set at False unless indicated to change
        is_checkpoint_step = False
        is_validation_step = False
        # Check if the current epoch is a validation or checkpoint step
        if (epoch > 0) and ((epoch + 1) != self.epochs):
            if (epoch + 1) % self.SAVE_MODEL_EPOCH_NUM == 0:
                is_checkpoint_step = True
            if (epoch + 1) % self.VALIDATION_EPOCH_NUM == 0:
                is_validation_step = True

        return is_checkpoint_step, is_validation_step

    def run_batches(self, 
                    data,
                    epoch,
                    train_dev_test = 'train'
                    ):
        if train_dev_test == 'train':
            total_samples = data.n_train_data_set
        elif train_dev_test == 'dev':
            total_samples = data.n_dev_data_set
        elif train_dev_test == 'test':
            predictions = []
            total_samples = data.n_prediction_data_set
        else:
            raise ValueError('wrong train_dev_test!')
        
        n_batches_per_epoch = total_samples//data.batch_size + 1

        total_training_loss = 0
        total_accuracy = 0

        for i in range(n_batches_per_epoch):
            batch_inputs, batch_targets, batch_seq_lens = data.next_batch(train_dev_test)

            feeds = {self.input_tensor: batch_inputs,
                     self.target: batch_targets,
                     self.seq_length: batch_seq_lens}

            feeds_pred = {self.input_tensor: batch_inputs,
                          self.seq_length: batch_seq_lens}

            if train_dev_test == 'train':
                batch_loss, _ = self.sess.run([self.avg_loss, self.optimizer], feed_dict = feeds)
                total_training_loss += batch_loss * data.batch_size * data.segment_length
                logger.debug('Avg batch cost: %.2f | Total train cost so far: %.2f', 
                              batch_loss, 
                              total_training_loss)

            if train_dev_test == 'train' or train_dev_test == 'dev':
                accuracy, label_pred, summary_line = self.sess.run([self.accuracy, self.label_pred, self.summary_op], 
                                                                    feed_dict = feeds)
                total_accuracy += accuracy * sum(batch_seq_lens)
                logger.debug('Prediction accuracy on the batch: %.2f', accuracy)
                if i % 4 == 0:
                    print('============== confusion matrix ===============')
                    # print(np.argmax(batch_targets.reshape(-1, self.n_character), 1))
                    # print(label_pred)
                    print(confusion_matrix(np.argmax(batch_targets.reshape(-1, self.n_character), 1), label_pred))
                    print('===============================================')
            else:
                label_pred = self.sess.run(self.label_pred, 
                                           feed_dict = feeds_pred)
                predictions.append(label_pred)
        if train_dev_test == 'test':
            return predictions

        epoch_accuracy = total_accuracy / total_samples / data.segment_length
        self.writer.add_summary(summary_line, epoch)

        return total_training_loss, epoch_accuracy

# to run in console
if __name__ == '__main__':
    import click
    # Use click to parse command line arguments
    @click.command()
    @click.option('--train_or_predict', type=bool, default=True, help='Train the model or predict model based on input')
    @click.option('--config', default='configs/neural_network.ini', help='Configuration file name')
    @click.option('--name', default=None, help='Path for retored model')
    @click.option('--train_from_model', type=bool, default=False, help='train from restored model')

    # for prediction
    @click.option('--test_data', default='data/test/', help='test data path')

    # Train RNN model using a given configuration file
    def main(config='configs/neural_network.ini',
             name = None,
             train_from_model = False,
             train_or_predict = True,
             test_data = None):

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        global logger
        logger = logging.getLogger(os.path.basename(__file__))

        # create the Tf_train_ctc class
        tf_train = trainRNN(conf_path=config,
                            model_name=name, 
                            predict = not train_or_predict)
        
        if train_or_predict:
            # run the training
            tf_train.run_model(train_from_model = train_from_model)
        else:
            pred = tf_train.run_model(train_from_model = False,
                                      prediction_data_dir = test_data)

    main()













