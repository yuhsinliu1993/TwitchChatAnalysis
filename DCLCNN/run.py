import numpy as np
import tensorflow as tf
import os
import sys
import argparse
import datetime
from six.moves import xrange  # pylint: disable=redefined-builtin
from utils import to_categorical, get_conv_shape
from input_handler import get_input_data_from_csv, batch_generator, get_input_data_from_text
from dclcnn import DCLCNN
from Layers import ConvBlockLayer

from keras.models import Model
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Dropout, Lambda
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint


tf.logging.set_verbosity(tf.logging.INFO)
# Basic model parameters as external flags.
FLAGS = None


def build_model(num_filters, num_classes, sequence_max_length=512, num_quantized_chars=71, embedding_size=16, learning_rate=0.001, top_k=3):
    inputs = Input(shape=(sequence_max_length, ), dtype='int32', name='inputs')
    embedded_sent = Embedding(num_quantized_chars, embedding_size, input_length=sequence_max_length)(inputs)

    # First conv layer
    conv = Conv1D(filters=64, kernel_size=3, strides=2, padding="same")(embedded_sent)

    # Each ConvBlock with one MaxPooling Layer
    for i in range(len(num_filters)):
        conv = ConvBlockLayer(get_conv_shape(conv), num_filters[i])(conv)
        conv = MaxPooling1D(pool_size=3, strides=2, padding="same")(conv)

    # k-max pooling (Finds values and indices of the k largest entries for the last dimension)
    def _top_k(x):
        x = tf.transpose(x, [0, 2, 1])
        k_max = tf.nn.top_k(x, k=top_k)
        return tf.reshape(k_max[0], (-1, num_filters[-1] * top_k))
    k_max = Lambda(_top_k, output_shape=(num_filters[-1] * top_k,))(conv)

    # 3 fully-connected layer with dropout regularization
    fc1 = Dropout(0.2)(Dense(512, activation='relu', kernel_initializer='he_normal')(k_max))
    fc2 = Dropout(0.2)(Dense(512, activation='relu', kernel_initializer='he_normal')(fc1))
    fc3 = Dense(num_classes, activation='softmax')(fc2)

    # define optimizer
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=False)

    model = Model(inputs=inputs, outputs=fc3)
    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])

    if FLAGS.load_model is not None:
        model.load_weights(FLAGS.load_model)

    return model


def train_sentiment(input_file, max_feature_length, n_class, embedding_size, learning_rate, batch_size, num_epochs, save_dir=None):
    # Stage 1: Convert raw texts into char-ids format && convert labels into one-hot vectors
    X_train, y_train_sentiment, _ = get_input_data_from_csv(input_file, max_feature_length)
    y_train_sentiment = to_categorical(y_train_sentiment, n_class)

    # Stage 2: Build Model
    num_filters = [64, 128, 256, 512]
    # dclcnn = DCLCNN2(num_filters=num_filters, num_classes=n_class, embedding_size=embedding_size, learning_rate=learning_rate)

    model = build_model(num_filters=num_filters, num_classes=n_class, embedding_size=embedding_size, learning_rate=learning_rate)

    # Stage 3: Training
    save_dir = save_dir if save_dir is not None else 'checkpoints'
    filepath = os.path.join(save_dir, "weights-{epoch:02d}-{val_acc:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    if FLAGS.verbose:
        print(model.summary())

    model.fit(
        x=X_train,
        y=y_train_sentiment,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.33,
        callbacks=[checkpoint],
        shuffle=True,
        verbose=FLAGS.verbose
    )


def train(input_file, max_feature_length):
    X_train, y_train_sentiment, y_train_comment = get_input_data_from_csv(input_file, max_feature_length)
    y_train_sentiment = to_categorical(y_train_sentiment, FLAGS.n_sentiment_classes)
    y_train_comment = to_categorical(y_train_comment, FLAGS.n_comment_classes)

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(X_train.shape[0]))
    X_shuffled = X_train[shuffle_indices]
    y_sentiment_shuffled = y_train_sentiment[shuffle_indices]
    y_comment_shuffled = y_train_comment[shuffle_indices]

    # Split train/test set
    slice = int(X_shuffled.shape[0] * 0.25)  # 1/4 for dev
    X_train, X_dev = X_shuffled[:-slice], X_shuffled[-slice:]
    y_train_sentiment, y_dev_sentiment = y_sentiment_shuffled[:-slice], y_sentiment_shuffled[-slice:]
    y_train_comment, y_dev_comment = y_comment_shuffled[:-slice], y_comment_shuffled[-slice:]

    print("Train/Dev size: %d / %d\n" % (y_train_sentiment.shape[0], y_dev_sentiment.shape[0]))

    print("[*] Start training sentiments ...")
    sentiment_eval_dict = {'X_dev': X_dev, 'y_dev': y_dev_sentiment}
    _train_sentiment(X_train, y_train_sentiment, sentiment_eval_dict)

    print("[*] Start training comments ...")
    comment_eval_dict = {'X_dev': X_dev, 'y_dev': y_dev_comment}
    _train_comment(X_train, y_train_comment, comment_eval_dict)


def _train_step(X_batch, y_batch, sess, model, summary_op, summary_writer):
    feed_dict = {
        model.X_input: X_batch,
        model.y_input: y_batch,
        model.dropout_keep_prob: 0.75
    }

    _, step, summaries, loss, accuracy = sess.run([model.train_op, model.global_step, summary_op, model.losses, model.accuracy], feed_dict)

    # Write fewer training summaries, to keep events file from growing so big.
    time_str = datetime.datetime.now().isoformat()
    if step % (FLAGS.evaluate_every / 2) == 0:
        print("{}: step: {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        summary_writer.add_summary(summaries, step)


def do_eval(eval_data, sess, model, summary_op, summary_writer):
    if not isinstance(eval_data, dict):
        raise TypeError('eval_data should be `dict` type. Your type of eval_data is %s' % type(eval_data))

    dev_num_batches = eval_data['X_dev'].shape[0] // FLAGS.batch_size + 1
    avg_acc = 0.0
    avg_loss = 0.0

    for i in xrange(dev_num_batches):
        X_batch = eval_data['X_dev'][i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]
        y_batch = eval_data['y_dev'][i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]

        feed_dict = {
            model.X_input: X_batch,
            model.y_input: y_batch,
            model.dropout_keep_prob: 1.0
        }
        step, summaries, loss, accuracy = sess.run([model.global_step, summary_op, model.losses, model.accuracy], feed_dict)
        avg_acc += accuracy / dev_num_batches
        avg_loss += loss / dev_num_batches

        summary_writer.add_summary(summaries, step)

        time_str = datetime.datetime.now().isoformat()
        print("batch " + str(i + 1) + " in dev ==>" + " {}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))

    print("\n---------------")
    print("Average accuracy = %.4f" % avg_acc)
    print("Average loss = %.4f " % avg_loss)
    print("---------------\n")


def _train_sentiment(X_train, y_train, eval_data):
    with tf.Graph().as_default():
        sess = tf.Session()

        with sess.as_default():
            dclcnn = DCLCNN(X_train, y_train, num_classes=FLAGS.n_sentiment_classes, embedding_size=FLAGS.embedding_size, l2_reg_weight_decay=FLAGS.l2_weight_decay, mode=FLAGS.mode)
            dclcnn.build_graph()

            print("[*] Writing to %s/sentiment/\n" % FLAGS.output_dir)
            train_summary_op = tf.summary.merge([dclcnn.grad_summaries, dclcnn.acc_summary, dclcnn.loss_summary])
            train_summary_dir = os.path.join(FLAGS.output_dir, "sentiment", "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            dev_summary_op = tf.summary.merge([dclcnn.loss_summary, dclcnn.acc_summary])
            dev_summary_dir = os.path.join(FLAGS.output_dir, "sentiment", "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            checkpoint_dir = os.path.join(FLAGS.output_dir, "sentiment", "checkpoints")
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                tf.logging.info('Loading checkpoint %s', ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                tf.logging.info('Create new session')
                # Initialize all variables
                sess.run(tf.global_variables_initializer())

            training_samples = X_train.shape[0]
            num_batches_per_epoch = training_samples // FLAGS.batch_size + 1
            batches = batch_generator(X_train, y_train, FLAGS.batch_size, FLAGS.num_epochs)

            for batch in batches:
                X_batch, y_batch = zip(*batch)
                _train_step(X_batch, y_batch, sess, dclcnn, train_summary_op, train_summary_writer)
                current_step = tf.train.global_step(sess, dclcnn.global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    print("\nSentiment Evaluation:")
                    do_eval(eval_data, sess, dclcnn, dev_summary_op, dev_summary_writer)

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def _train_comment(X_train, y_train, eval_data):
    with tf.Graph().as_default():
        sess = tf.Session()

        with sess.as_default():
            dclcnn = DCLCNN(X_train, y_train, num_classes=FLAGS.n_comment_classes, embedding_size=FLAGS.embedding_size, l2_reg_weight_decay=FLAGS.l2_weight_decay, mode=FLAGS.mode)
            dclcnn.build_graph()

            print("[*] Writing to %s/comment/\n" % FLAGS.output_dir)
            train_summary_op = tf.summary.merge([dclcnn.grad_summaries, dclcnn.acc_summary, dclcnn.loss_summary])
            train_summary_dir = os.path.join(FLAGS.output_dir, "comment", "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            dev_summary_op = tf.summary.merge([dclcnn.loss_summary, dclcnn.acc_summary])
            dev_summary_dir = os.path.join(FLAGS.output_dir, "comment", "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            checkpoint_dir = os.path.join(FLAGS.output_dir, "comment", "checkpoints")
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                tf.logging.info('Loading checkpoint %s', ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                tf.logging.info('Create new session')
                # Initialize all variables
                sess.run(tf.global_variables_initializer())

            training_samples = X_train.shape[0]
            num_batches_per_epoch = training_samples // FLAGS.batch_size + 1
            batches = batch_generator(X_train, y_train, FLAGS.batch_size, FLAGS.num_epochs)

            for batch in batches:
                X_batch, y_batch = zip(*batch)
                _train_step(X_batch, y_batch, sess, dclcnn, train_summary_op, train_summary_writer)
                current_step = tf.train.global_step(sess, dclcnn.global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    print("\nComment Evaluation:")
                    do_eval(eval_data, sess, dclcnn, dev_summary_op, dev_summary_writer)

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def inference(test_file, max_feature_length):
    comments = []
    sentiments = []
    comment_classes = []

    with open(test_file, 'r') as f:
        for line in f.readlines():
            comment, sentiment, _class = line.split(',')
            comments.append(comment)
            sentiments.append(sentiment)
            comment_classes.append(_class)

    for i in xrange(len(comments)):
        X, y_sentiment, y_comment = get_input_data_from_text(comments[i], sentiments[i], comment_classes[i], max_feature_length)
        y_sentiment = to_categorical(y_sentiment, FLAGS.n_sentiment_classes)
        y_comment = to_categorical(y_comment, FLAGS.n_comment_classes)

        _infer_sentiment(X, y_sentiment, comments[i])
        _infer_comment(X, y_comment, comments[i])


def _infer_sentiment(X, y, comment):
    with tf.Graph().as_default():
        sess = tf.Session()

        with sess.as_default():
            dclcnn = DCLCNN(X, y, num_classes=FLAGS.n_sentiment_classes, embedding_size=FLAGS.embedding_size, l2_reg_weight_decay=FLAGS.l2_weight_decay, mode=FLAGS.mode)
            dclcnn.build_graph()

            checkpoint_dir = os.path.join(FLAGS.output_dir, "sentiment", "checkpoints")
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                tf.logging.info('Loading checkpoint %s', ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise ValueError("Cannot load checkpoint.")

            prediction = sess.run([dclcnn.predictions], feed_dict={dclcnn.X_input: X, dclcnn.y_input: y})[0]
            pred = int(np.argmax(prediction, axis=1)) - 1  # shift back to (-1, 0, 1) classes
            y_true = int(np.argmax(y, axis=1)) - 1  # shift back to (-1, 0, 1) classes

            print("---------- SENTIMENT INFERENCE ----------")
            print("COMMENT: " + comment)
            print("[*] SENTIMENT Y_TRUE: %d" % y_true)
            print("[*] SENTIMENT PREDICTION: %d\n" % pred)


def _infer_comment(X, y, comment):
    with tf.Graph().as_default():
        sess = tf.Session()

        with sess.as_default():
            dclcnn = DCLCNN(X, y, num_classes=FLAGS.n_comment_classes, embedding_size=FLAGS.embedding_size, l2_reg_weight_decay=FLAGS.l2_weight_decay, mode=FLAGS.mode)
            dclcnn.build_graph()

            checkpoint_dir = os.path.join(FLAGS.output_dir, "comment", "checkpoints")
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                tf.logging.info('Loading checkpoint %s', ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise ValueError("Cannot load checkpoint.")

            prediction = sess.run([dclcnn.predictions], feed_dict={dclcnn.X_input: X, dclcnn.y_input: y})[0]
            pred = np.argmax(prediction, axis=1)
            y_true = np.argmax(y, axis=1)

            print("---------- COMMENT INFERENCE ----------")
            print("COMMENT: " + comment)
            print("[*] COMMENT Y_TRUE: %d" % y_true)
            print("[*] COMMENT PREDICTION: %d\n" % pred)


def run(_):
    if FLAGS.mode == 'train':
        # train(FLAGS.input_data, FLAGS.max_feature_length)
        train_sentiment(input_file=FLAGS.input_data, max_feature_length=FLAGS.max_feature_length, n_class=FLAGS.n_sentiment_classes, embedding_size=FLAGS.embedding_size, learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)
    elif FLAGS.mode == 'eval':
        pass
    elif FLAGS.mode == 'infer':
        inference(FLAGS.test_data, FLAGS.max_feature_length)
    else:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_sentiment_classes',
        type=int,
        default=3,
        help='Specify number of classes of sentiments'
    )
    parser.add_argument(
        '--n_comment_classes',
        type=int,
        default=10,
        help='Specify number of classes of comments'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10,
        help='Specify number of epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size. Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Specify learning rate'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='sgd',
        help='Specify optimizer'
    )
    parser.add_argument(
        '--input_data',
        type=str,
        default='./data/train.csv',
        help='Location store the input data (only accept `csv` format)'  # [TODO: to support more data formats]
    )
    parser.add_argument(
        '--test_data',
        type=str,
        default='./data/test.txt',
        help='Specify test data path'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/Users/Michaeliu/Twitch/DCLCNN/TRAIN_MODEL',
        help='Directory to store the summaries and checkpoints.'
    )
    parser.add_argument(
        '--streamer',
        type=str,
        default='thijs',
        help='Specify a twitch streamer'
    )
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=16,
        help='Specify embedding size'
    )
    parser.add_argument(
        '--max_feature_length',
        type=int,
        default=512,
        help='Specify max feature length'
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=50,
        help='do evaluation after # numbers of training steps'
    )
    parser.add_argument(
        '--checkpoint_every',
        type=int,
        default=50,
        help='Save checkpoint after # numbers of training steps'
    )
    parser.add_argument(
        '--l2_weight_decay',
        type=float,
        default=1e-3,
        help='Specify max feature length'
    )
    parser.add_argument(
        '--mode',
        type=str,
        help='Specify mode: `train` or `eval` or `infer`',
        required=True
    )
    parser.add_argument(
        '--load_model',
        type=str,
        help='Specify the location of model weights',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose on training',
        default=False
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=run, argv=[sys.argv[0]] + unparsed)
