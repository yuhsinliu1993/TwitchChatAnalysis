import tensorflow as tf
import os
import sys
import argparse
from utils import to_categorical, get_conv_shape
from input_handler import get_input_data_from_csv, get_input_data_from_text
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


def build_model(num_filters, num_classes, sequence_max_length=128, num_quantized_chars=71, embedding_size=16, learning_rate=0.001, top_k=2):
    inputs = Input(shape=(sequence_max_length, ), dtype='int32', name='inputs')
    embedded_sent = Embedding(num_quantized_chars, embedding_size, input_length=sequence_max_length)(inputs)

    # First conv layer
    conv = Conv1D(filters=8, kernel_size=2, strides=1, padding="same")(embedded_sent)

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
    fc1 = Dropout(0.7)(Dense(128, activation='relu', kernel_initializer='he_normal')(k_max))
    fc2 = Dropout(0.7)(Dense(128, activation='relu', kernel_initializer='he_normal')(fc1))
    fc3 = Dense(num_classes, activation='softmax')(fc2)

    # define optimizer
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=False)

    model = Model(inputs=inputs, outputs=fc3)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    if FLAGS.load_model is not None:
        model.load_weights(FLAGS.load_model)

    return model


def train_sentiment(input_file, max_feature_length, n_class, embedding_size, learning_rate, batch_size, num_epochs, save_dir=None, print_summary=False):
    # Stage 1: Convert raw texts into char-ids format && convert labels into one-hot vectors
    X_train, y_train_sentiment, _ = get_input_data_from_csv(input_file, max_feature_length)
    y_train_sentiment = to_categorical(y_train_sentiment, n_class)

    # Stage 2: Build Model
    model = build_model(num_filters=[16, 32, 64, 128], num_classes=n_class, sequence_max_length=FLAGS.max_feature_length, embedding_size=embedding_size, learning_rate=learning_rate)

    # Stage 3: Training
    save_dir = save_dir if save_dir is not None else 'checkpoints'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, "weights-{epoch:02d}-{val_acc:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')

    if print_summary:
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


def do_inference(test_data, max_feature_length):
    pass


def run(_):
    if FLAGS.mode == 'train':
        train_sentiment(input_file=FLAGS.input_data, max_feature_length=FLAGS.max_feature_length, n_class=FLAGS.n_sentiment_classes, embedding_size=FLAGS.embedding_size, learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs, save_dir=FLAGS.save_dir, print_summary=FLAGS.print_summary)
    elif FLAGS.mode == 'eval':
        pass
    elif FLAGS.mode == 'infer':
        do_inference(FLAGS.test_data, FLAGS.max_feature_length)
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
        '--save_dir',
        type=str,
        default='checkpoints',
        help='Specify checkpoint directory'
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
        default=128,
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
    parser.add_argument(
        '--print_summary',
        action='store_true',
        help='Print out model summary',
        default=False
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=run, argv=[sys.argv[0]] + unparsed)
