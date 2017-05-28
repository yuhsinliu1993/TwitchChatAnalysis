import re
import tensorflow as tf


class DCLCNN(object):
    """
    Implementation of `Deep Character-Level Convolutional Neural Network` Model based on "Very Deep Convolutional Networks for Natural Language Processing" proposed by FACEBOOK AI RESEARCH TEAM.
    See https://arxiv.org/abs/1606.01781 for more infomations.
    """

    def __init__(self, X_input, y_input, mode, num_classes, filter_size=3, l2_reg_weight_decay=0.001, sequence_max_length=512, num_quantized_chars=71, embedding_size=16):
        self.X_input = X_input
        self.y_input = y_input
        self.mode = mode
        self.num_classes = num_classes
        self.filter_size = filter_size
        self.l2_reg_weight_decay = l2_reg_weight_decay
        self.sequence_max_length = sequence_max_length
        self.num_quantized_chars = num_quantized_chars
        self.embedding_size = embedding_size

        self._extra_train_ops = []

    def build_graph(self):
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.l2_loss = tf.constant(0.0)

        self._build_model()

        if self.mode == 'train':
            self._build_train_op()

    def _build_train_op(self):
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.losses)
        train_op = optimizer.apply_gradients(grads_and_vars,
                                             global_step=self.global_step,
                                             name="train_step")

        train_ops = [train_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

        self.grad_summaries = []
        for grad, var in grads_and_vars:
            if grad is not None:
                name = re.sub(r':', '_', var.name)
                grad_hist_summary = tf.summary.histogram("{}/grad/histogram".format(name), grad)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(name), tf.nn.zero_fraction(grad))
                self.grad_summaries.append(grad_hist_summary)
                self.grad_summaries.append(sparsity_summary)

    def _build_model(self):
        self.X_input = tf.placeholder(tf.int32, [None, self.sequence_max_length], name="X_input")  # # X_input: must be converted into "ids"
        self.y_input = tf.placeholder(tf.float32, [None, self.num_classes], name="y_input")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedding_W = tf.get_variable("weights", shape=[self.num_quantized_chars, self.embedding_size], initializer=tf.random_uniform_initializer(-1.0, 1.0), dtype=tf.float32)  # shape=(71, 16)
            self.embedded_sent = tf.nn.embedding_lookup(params=embedding_W, ids=self.X_input, name="embedding_lookup")  # shape=(?, 512, 16)
            self.embedded_sent_expanded = tf.expand_dims(self.embedded_sent, -1)  # expand one dim on the end. shape=(?, 512, 16, 1) which is (batch_size, input_length(height), embedding_size(width), in_channels)

        with tf.name_scope("layer0"):
            filter_0 = [3, self.embedding_size, 1, 64]  # [filter_height, filter_width, in_channels, out_channels]
            strides_0 = [1, 1, self.embedding_size, 1]  # []
            self.conv0 = self._conv_layer(self.embedded_chars_expanded, filter_0, strides_0, 'layer0')

        with tf.name_scope("conv_block"):
            self.conv1 = self._convolutional_block(self.conv0, 64, 'block_1')
            self.conv2 = self._convolutional_block(self.conv1, 128, 'block_2')
            self.conv3 = self._convolutional_block(self.conv2, 256, 'block_3')
            self.conv4 = self._convolutional_block(self.conv3, 512, 'block_4')
            self.conv5 = tf.transpose(self.conv4, [0, 3, 2, 1])

        with tf.name_scope("maxpooling"):
            self.pooling = tf.nn.top_k(self.conv5, k=2, name='k-maxpooling')
            self.pooling = tf.reshape(self.pooling[0], (-1, 512 * 2))

        with tf.name_scope("fc"):
            self.fc1 = self._fully_connected_layer(self.pooling, 512, scope='fc1', stddev=0.1)
            self.fc1 = tf.nn.relu(self.fc1)

            self.fc2 = self._fully_connected_layer(self.fc1, 512, scope='fc2', stddev=0.1)
            self.fc2 = tf.nn.relu(self.fc2)

            self.logits = self._fully_connected_layer(self.fc2, self.num_classes, scope='logits', stddev=0.1)

        self.predictions = tf.nn.softmax(logits=self.logits)
        self.losses = self._loss()
        self.accuracy = self._accuracy()

    def _fully_connected_layer(self, input_tensor, output_dim, stddev=0.1, scope=None):
        with tf.variable_scope(scope or 'fc_layer'):
            W = tf.get_variable('weights', [input_tensor.get_shape()[1], output_dim], initializer=tf.random_normal_initializer(stddev=stddev))
            b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            return tf.matmul(input_tensor, W) + b

    def _convolutional_block(self, input_tensor, filter_num, scope):
        with tf.variable_scope(scope):
            filter_shape1 = [3, 1, input_tensor.get_shape()[3], filter_num]  # [filter_height, filter_width, in_channels, out_channels]
            filter_1 = tf.get_variable('filter1', filter_shape1, initializer=tf.random_normal_initializer(stddev=0.05))
            conv1 = tf.nn.conv2d(input_tensor, filter_1, strides=[1, 1, filter_shape1[1], 1], padding="SAME")
            batch_norm1 = tf.layers.batch_normalization(conv1, name="%s_BN1" % scope)
            # batch_norm1 = tflearn.layers.normalization.batch_normalization(conv1, scope="%s_BN1" % scope)

            filter_shape2 = [3, 1, batch_norm1.get_shape()[3], filter_num]
            filter_2 = tf.get_variable('fileer2', filter_shape2, initializer=tf.random_normal_initializer(stddev=0.05))
            conv2 = tf.nn.conv2d(tf.nn.relu(batch_norm1), filter_2, strides=[1, 1, filter_shape2[1], 1], padding="SAME")
            batch_normal2 = tf.layers.batch_normalization(conv2, name="%s_BN2" % scope)
            # batch_normal2 = tflearn.layers.normalization.batch_normalization(conv2, scope="%s_BN2" % scope)

            return tf.nn.max_pool(tf.nn.relu(batch_normal2), ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name="pool1")

    def _conv_layer(self, input_tensor, filter_shape, strides, scope):
        """
        Computes a 2-D convolution given 4-D input and filter tensors

        Given an input tensor of shape [batch, in_height, in_width, in_channels] and a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]

        INPUTS:
            input_tensor: shape=()
        """
        with tf.variable_scope(scope):
            filter_1 = tf.get_variable('filter1', filter_shape, initializer=tf.random_normal_initializer(stddev=0.05))
            conv = tf.nn.conv2d(input_tensor, filter=filter_1, strides=strides, padding="SAME")
            conv = tf.layers.batch_normalization(conv, name="%s_BN" % scope)

            return conv

    def _loss(self):
        with tf.variable_scope('losses'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_input)
            losses = tf.reduce_mean(cross_entropy, name='corss_entropy')
            losses += self._l2_decay()

            self.loss_summary = tf.summary.scalar('losses', losses)

            return losses

    def _l2_decay(self):
        losses = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'weights') > 0:
                losses.append(tf.nn.l2_loss(var))

        return tf.multiply(self.l2_reg_weight_decay, tf.add_n(losses))

    def _accuracy(self):
        with tf.name_scope("accuracy"):
            y_true = tf.argmax(self.y_input, axis=1)
            correct_predictions = tf.argmax(self.predictions, axis=1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(correct_predictions, y_true), tf.float32), name="acc")

            self.acc_summary = tf.summary.scalar("accuracy", accuracy)

            return accuracy
