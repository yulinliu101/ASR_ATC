# Note: All calls to tf.name_scope or tf.summary.* support TensorBoard visualization.

import os
import tensorflow as tf
from configparser import ConfigParser

# from models.RNN.utils import variable_on_gpu

def variable_on_cpu(name, shape, initializer):
    """
    Next we concern ourselves with graph creation.
    However, before we do so we must introduce a utility function ``variable_on_gpu()``
    used to create a variable in CPU memory.
    """
    # Use the /cpu:0 device for scoped operations
    with tf.device('/cpu:0'):
        # Create or get apropos variable
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var

def variable_on_gpu(name, shape, initializer):
    """
    Next we concern ourselves with graph creation.
    However, before we do so we must introduce a utility function ``variable_on_gpu()``
    used to create a variable in CPU memory.
    """
    # Use the /cpu:0 device for scoped operations
    with tf.device('/device:GPU:0'):
        # Create or get apropos variable
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var

def create_optimizer():
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon)
    return optimizer

def BiRNN(conf_path, batch_x, seq_length, n_input, n_context):
    """
    This function was initially based on open source code from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/DeepSpeech.py

    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    """
    parser = ConfigParser(os.environ)
    parser.read(conf_path)
    n_character = parser.getint('birnn', 'n_character')

    dropout = [0.1, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
    relu_clip = parser.getint('birnn', 'relu_clip')

    b1_stddev = parser.getfloat('birnn', 'b1_stddev')
    h1_stddev = parser.getfloat('birnn', 'h1_stddev')
    
    b2_stddev = parser.getfloat('birnn', 'b2_stddev')
    h2_stddev = parser.getfloat('birnn', 'h2_stddev')

    b3_stddev = parser.getfloat('birnn', 'b3_stddev')
    h3_stddev = parser.getfloat('birnn', 'h3_stddev')

    b4_stddev = parser.getfloat('birnn', 'b4_stddev')
    h4_stddev = parser.getfloat('birnn', 'h4_stddev')
    
    b5_stddev = parser.getfloat('birnn', 'b5_stddev')
    h5_stddev = parser.getfloat('birnn', 'h5_stddev')

    b_voc_stddev = parser.getfloat('birnn', 'b_voc_stddev')
    h_voc_stddev = parser.getfloat('birnn', 'h_voc_stddev')

    n_hidden_1 = parser.getint('birnn', 'n_hidden_1')
    n_hidden_2 = parser.getint('birnn', 'n_hidden_2')
    n_cell_dim = parser.getint('birnn', 'n_cell_dim')
    n_hidden_3 = int(eval(parser.get('birnn', 'n_hidden_3')))
    n_hidden_4 = parser.getint('birnn', 'n_hidden_4')
    n_hidden_5 = parser.getint('birnn', 'n_hidden_5')
    n_hidden_voc = parser.getint('birnn', 'n_hidden_voc')
    
    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    # n_input is the # of (original) features per frame: default to be 26
    batch_x_shape = tf.shape(batch_x)

    # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
    # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

    # Permute n_steps and batch_size
    batch_x = tf.transpose(batch_x, [1, 0, 2])
    # Reshape to prepare input for first layer
    batch_x = tf.reshape(batch_x,
                         [-1, n_input + 2 * n_input * n_context])  # (n_steps*batch_size, n_input + 2*n_input*n_context)

    # The next three blocks will pass `batch_x` through three hidden layers with
    # clipped RELU activation and dropout.

    # 1st layer
    with tf.name_scope('embedding'):
        b1 = variable_on_gpu('b1', [n_hidden_1], tf.random_normal_initializer(stddev=b1_stddev))
        h1 = variable_on_gpu('h1', [n_input + 2 * n_input * n_context, n_hidden_1],
                             tf.random_normal_initializer(stddev=h1_stddev))
        layer_emb = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), relu_clip)
        layer_emb = tf.nn.dropout(layer_emb, (1.0 - dropout[0]))

        with tf.device('/cpu:0'):
            tf.summary.histogram("weights", h1)
            tf.summary.histogram("biases", b1)
            tf.summary.histogram("activations", layer_emb)

    with tf.name_scope('fc1'):
        b2 = variable_on_gpu('b2', [n_hidden_2], tf.random_normal_initializer(stddev=b2_stddev))
        h2 = variable_on_gpu('h2', [n_hidden_1, n_hidden_2],
                             tf.random_normal_initializer(stddev=h2_stddev))
        layer_fc2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_emb, h2), b2)), relu_clip)
        layer_fc2 = tf.nn.dropout(layer_fc2, (1.0 - dropout[1]))

        with tf.device('/cpu:0'):

            tf.summary.histogram("weights", h2)
            tf.summary.histogram("biases", b2)
            tf.summary.histogram("activations", layer_fc2)


    # # debug layer
    # with tf.name_scope('debug'):
    #     b_tmp = variable_on_gpu('b_tmp', [(2 * n_cell_dim)], tf.random_normal_initializer(stddev=0.05))
    #     h_tmp = variable_on_gpu('h_tmp', [n_hidden_2, (2 * n_cell_dim)],
    #                          tf.random_normal_initializer(stddev=0.05))
    #     outputs = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_fc2, h_tmp), b_tmp)), relu_clip)
    #     outputs = tf.nn.dropout(outputs, (1.0 - 0.25))

    #     tf.summary.histogram("weights", b_tmp)
    #     tf.summary.histogram("biases", h_tmp)
    #     tf.summary.histogram("activations", outputs)

    # Create the forward and backward LSTM units. Inputs have length `n_cell_dim`.
    # LSTM forget gate bias initialized at `1.0` (default), meaning less forgetting
    # at the beginning of training (remembers more previous info)
    with tf.name_scope('bidirectional_lstm'):
        # Forward direction cell:
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,
                                                     input_keep_prob=1.0 - dropout[2],
                                                     output_keep_prob=1.0 - dropout[2],
                                                     # seed=random_seed,
                                                     )
        # Backward direction cell:
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,
                                                     input_keep_prob=1.0 - dropout[3],
                                                     output_keep_prob=1.0 - dropout[3],
                                                     # seed=random_seed,
                                                     )

        # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
        # as the LSTM BRNN expects its input to be of shape `[max_time, batch_size, input_size]`.
        layer_fc2 = tf.reshape(layer_fc2, [-1, batch_x_shape[0], n_hidden_2])

        # Now we feed `layer_3` into the LSTM BRNN cell and obtain the LSTM BRNN output.
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                 cell_bw=lstm_bw_cell,
                                                                 inputs=layer_fc2,
                                                                 dtype=tf.float32,
                                                                 time_major=True,
                                                                 sequence_length=seq_length)
        with tf.device('/cpu:0'):
            tf.summary.histogram("activations", outputs)

        # Reshape outputs from two tensors each of shape [n_steps, batch_size, n_cell_dim]
        # to a single tensor of shape [n_steps*batch_size, 2*n_cell_dim]
        outputs = tf.concat(outputs, 2)
        outputs = tf.reshape(outputs, [-1, 2 * n_cell_dim])

    with tf.name_scope('fc2'):
        # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
        b3 = variable_on_gpu('b3', [n_hidden_3], tf.random_normal_initializer(stddev=b3_stddev))
        h3 = variable_on_gpu('h3', [(2 * n_cell_dim), n_hidden_3], tf.random_normal_initializer(stddev=h3_stddev))
        layer_fc3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h3), b3)), relu_clip)
        layer_fc3 = tf.nn.dropout(layer_fc3, (1.0 - dropout[4]))
        with tf.device('/cpu:0'):
            tf.summary.histogram("weights", h3)
            tf.summary.histogram("biases", b3)
            tf.summary.histogram("activations", layer_fc3)

    with tf.name_scope('fc3'):
        # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
        b4 = variable_on_gpu('b4', [n_hidden_4], tf.random_normal_initializer(stddev=b4_stddev))
        h4 = variable_on_gpu('h4', [n_hidden_3, n_hidden_4], tf.random_normal_initializer(stddev=h4_stddev))
        layer_fc4 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_fc3, h4), b4)), relu_clip)
        layer_fc4 = tf.nn.dropout(layer_fc4, (1.0 - dropout[5]))
        with tf.device('/cpu:0'):
            tf.summary.histogram("weights", h4)
            tf.summary.histogram("biases", b4)
            tf.summary.histogram("activations", layer_fc4)

    with tf.name_scope('fc4'):
        # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
        b5 = variable_on_gpu('b5', [n_hidden_5], tf.random_normal_initializer(stddev=b5_stddev))
        h5 = variable_on_gpu('h5', [n_hidden_4, n_hidden_5], tf.random_normal_initializer(stddev=h5_stddev))
        layer_fc5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_fc4, h5), b5)), relu_clip)
        layer_fc5 = tf.nn.dropout(layer_fc5, (1.0 - dropout[6]))
        with tf.device('/cpu:0'):
            tf.summary.histogram("weights", h5)
            tf.summary.histogram("biases", b5)
            tf.summary.histogram("activations", layer_fc5)

    with tf.name_scope('vocab'):
        # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
        b_voc = variable_on_gpu('b_voc', [n_character], tf.random_normal_initializer(stddev=b_voc_stddev))
        h_voc = variable_on_gpu('h_voc', [n_hidden_5, n_character], tf.random_normal_initializer(stddev=h_voc_stddev))
        logits = tf.add(tf.matmul(layer_fc5, h_voc), b_voc)
        with tf.device('/cpu:0'):
            tf.summary.histogram("weights", h_voc)
            tf.summary.histogram("biases", b_voc)
            tf.summary.histogram("activations", logits)

    # Finally we reshape layer_vocab from a tensor of shape [n_steps*batch_size, n_hidden_4]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_4].
    # Note, that this differs from the input in that it is time-major.
    logits = tf.reshape(logits, [-1, batch_x_shape[0], n_character])

    # not time-major axis
    # before transose logits has the dimension of [n_steps*batch_size, n_hidden_4]
    logits = tf.transpose(logits, [1, 0, 2])
    # logits = tf.reshape(logits, [batch_x_shape[0], -1, n_character])
    with tf.device('/cpu:0'):
        summary_op = tf.summary.merge_all()

    # Output shape: [batch_size, n_steps, n_hidden_6]
    return logits, summary_op


def BiRNN_FC(conf_path, batch_x, seq_length, n_input, n_context):
    """
    This function was initially based on open source code from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/DeepSpeech.py

    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    """
    parser = ConfigParser(os.environ)
    parser.read(conf_path)

    dropout = [float(x) for x in parser.get('birnn_fc', 'dropout_rates').split(',')]
    relu_clip = parser.getint('birnn_fc', 'relu_clip')

    b1_stddev = parser.getfloat('birnn_fc', 'b1_stddev')
    h1_stddev = parser.getfloat('birnn_fc', 'h1_stddev')
    b2_stddev = parser.getfloat('birnn_fc', 'b2_stddev')
    h2_stddev = parser.getfloat('birnn_fc', 'h2_stddev')
    b3_stddev = parser.getfloat('birnn_fc', 'b3_stddev')
    h3_stddev = parser.getfloat('birnn_fc', 'h3_stddev')
    b5_stddev = parser.getfloat('birnn_fc', 'b5_stddev')
    h5_stddev = parser.getfloat('birnn_fc', 'h5_stddev')
    b6_stddev = parser.getfloat('birnn_fc', 'b6_stddev')
    h6_stddev = parser.getfloat('birnn_fc', 'h6_stddev')

    n_hidden_1 = parser.getint('birnn_fc', 'n_hidden_1')
    n_hidden_2 = parser.getint('birnn_fc', 'n_hidden_2')
    n_hidden_5 = parser.getint('birnn_fc', 'n_hidden_5')
    n_cell_dim = parser.getint('birnn_fc', 'n_cell_dim')

    n_hidden_3 = int(eval(parser.get('birnn_fc', 'n_hidden_3')))
    n_hidden_6 = parser.getint('birnn_fc', 'n_hidden_6')

    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    # n_input is the # of (original) features per frame: default to be 26
    batch_x_shape = tf.shape(batch_x)

    # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
    # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

    # Permute n_steps and batch_size
    batch_x = tf.transpose(batch_x, [1, 0, 2])
    # Reshape to prepare input for first layer
    batch_x = tf.reshape(batch_x,
                         [-1, n_input + 2 * n_input * n_context])  # (n_steps*batch_size, n_input + 2*n_input*n_context)

    # The next three blocks will pass `batch_x` through three hidden layers with
    # clipped RELU activation and dropout.

    # 1st layer
    with tf.name_scope('fc1'):
        b1 = variable_on_gpu('b1', [n_hidden_1], tf.random_normal_initializer(stddev=b1_stddev))
        h1 = variable_on_gpu('h1', [n_input + 2 * n_input * n_context, n_hidden_1],
                             tf.random_normal_initializer(stddev=h1_stddev))
        layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), relu_clip)
        layer_1 = tf.nn.dropout(layer_1, (1.0 - dropout[0]))

        tf.summary.histogram("weights", h1)
        tf.summary.histogram("biases", b1)
        tf.summary.histogram("activations", layer_1)

    # 2nd layer
    with tf.name_scope('fc2'):
        b2 = variable_on_gpu('b2', [n_hidden_2], tf.random_normal_initializer(stddev=b2_stddev))
        h2 = variable_on_gpu('h2', [n_hidden_1, n_hidden_2], tf.random_normal_initializer(stddev=h2_stddev))
        layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), relu_clip)
        layer_2 = tf.nn.dropout(layer_2, (1.0 - dropout[1]))

        tf.summary.histogram("weights", h2)
        tf.summary.histogram("biases", b2)
        tf.summary.histogram("activations", layer_2)

    # 3rd layer
    with tf.name_scope('fc3'):
        b3 = variable_on_gpu('b3', [n_hidden_3], tf.random_normal_initializer(stddev=b3_stddev))
        h3 = variable_on_gpu('h3', [n_hidden_2, n_hidden_3], tf.random_normal_initializer(stddev=h3_stddev))
        layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), relu_clip)
        layer_3 = tf.nn.dropout(layer_3, (1.0 - dropout[2]))

        tf.summary.histogram("weights", h3)
        tf.summary.histogram("biases", b3)
        tf.summary.histogram("activations", layer_3)

    # Create the forward and backward LSTM units. Inputs have length `n_cell_dim`.
    # LSTM forget gate bias initialized at `1.0` (default), meaning less forgetting
    # at the beginning of training (remembers more previous info)
    with tf.name_scope('bidirectional_lstm'):
        # Forward direction cell:
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,
                                                     input_keep_prob=1.0 - dropout[3],
                                                     output_keep_prob=1.0 - dropout[3],
                                                     # seed=random_seed,
                                                     )
        # Backward direction cell:
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,
                                                     input_keep_prob=1.0 - dropout[4],
                                                     output_keep_prob=1.0 - dropout[4],
                                                     # seed=random_seed,
                                                     )

        # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
        # as the LSTM BRNN expects its input to be of shape `[max_time, batch_size, input_size]`.
        layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], n_hidden_3])

        # Now we feed `layer_3` into the LSTM BRNN cell and obtain the LSTM BRNN output.
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                 cell_bw=lstm_bw_cell,
                                                                 inputs=layer_3,
                                                                 dtype=tf.float32,
                                                                 time_major=True,
                                                                 sequence_length=seq_length)

        tf.summary.histogram("activations", outputs)

        # Reshape outputs from two tensors each of shape [n_steps, batch_size, n_cell_dim]
        # to a single tensor of shape [n_steps*batch_size, 2*n_cell_dim]
        outputs = tf.concat(outputs, 2)
        outputs = tf.reshape(outputs, [-1, 2 * n_cell_dim])

    with tf.name_scope('fc5'):
        # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
        b5 = variable_on_gpu('b5', [n_hidden_5], tf.random_normal_initializer(stddev=b5_stddev))
        h5 = variable_on_gpu('h5', [(2 * n_cell_dim), n_hidden_5], tf.random_normal_initializer(stddev=h5_stddev))
        layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), relu_clip)
        layer_5 = tf.nn.dropout(layer_5, (1.0 - dropout[5]))

        tf.summary.histogram("weights", h5)
        tf.summary.histogram("biases", b5)
        tf.summary.histogram("activations", layer_5)

    with tf.name_scope('fc6'):
        # Now we apply the weight matrix `h6` and bias `b6` to the output of `layer_5`
        # creating `n_classes` dimensional vectors, the logits.
        b6 = variable_on_gpu('b6', [n_hidden_6], tf.random_normal_initializer(stddev=b6_stddev))
        h6 = variable_on_gpu('h6', [n_hidden_5, n_hidden_6], tf.random_normal_initializer(stddev=h6_stddev))
        layer_6 = tf.add(tf.matmul(layer_5, h6), b6)

        tf.summary.histogram("weights", h6)
        tf.summary.histogram("biases", b6)
        tf.summary.histogram("activations", layer_6)

    # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
    # Note, that this differs from the input in that it is time-major.
    layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], n_hidden_6])

    summary_op = tf.summary.merge_all()

    # Output shape: [n_steps, batch_size, n_hidden_6]
    return layer_6, summary_op
