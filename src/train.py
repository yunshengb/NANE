from __future__ import division
from __future__ import print_function

import time, os
import tensorflow as tf
import numpy as np

from utils import *
from models import NANE

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

current_folder = os.path.dirname(os.path.realpath(__file__))

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
# 'cora', 'blog', 'flickr'
flags.DEFINE_integer('debug', 1, '0: Normal; 1: Debug.')
flags.DEFINE_string('desc',
                    'emergent_dim_50',
                    'Description of the experiment.')
flags.DEFINE_integer('need_batch', 2, 'Need mini-batch or not.')
flags.DEFINE_string('device', 'gpu', 'cpu|gpu.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 100, 'Number of units in hidden layer 2.')
flags.DEFINE_float('train_ratio', 0.2, 'Ratio of training data.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('fs', 1, 'Fast sampling of higher-order neighbors.')
flags.DEFINE_integer('need_higher', 1, 'Need higher-order neighbors.')

# Load data
adj, features, y_train = load_data(FLAGS.dataset)
support = [laplacian(adj)]
num_supports = 1
model_func = NANE

# Define placeholders
N = get_shape(adj)[0]
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(
    num_supports)],
    'output_dim': get_shape(y_train),
}

if FLAGS.need_batch:
    placeholders['batch'] = tf.placeholder(tf.int32)
    placeholders['pos_labels'] = tf.placeholder(tf.int32)
    placeholders['neg_labels'] = tf.placeholder(tf.int32)
    placeholders['usl_labels'] = tf.placeholder( \
        tf.float32, shape=(None, 8 if FLAGS.need_higher else 6))
    placeholders['num_data'] = get_shape(adj)[0]
else:
    placeholders['usl_labels'] = tf.placeholder(tf.float32, shape=(N, N))
    placeholders['sims_mask'] = tf.placeholder(tf.float32,
                                               shape=(N, N))

# Create model
input_dim = features.shape[1] if features is not None else N
model = model_func(placeholders, input_dim=input_dim, logging=True)

# Initialize session
session_conf = tf.ConfigProto(
    device_count={'CPU': 1, 'GPU': 0},
    allow_soft_placement=True,
    log_device_placement=False
)

if FLAGS.device == 'cpu':
    sess = tf.Session(config=session_conf)
else:
    sess = tf.Session()


def need_print(epoch=None):
    if FLAGS.debug or not epoch:
        return False
    return epoch % 100 == 0


# Summary.
dir = prepare_exp_dir(FLAGS)

# Init variables
sess.run(tf.global_variables_initializer())

loss = np.inf

# Train model
for epoch in range(FLAGS.epochs):
    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj, features, support, y_train,
                                    placeholders, loss)
    #feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    if need_print(epoch):
        embeddings = model.layers[-1].embeddings
        print_var(embeddings,
                  'nane_%s_emb_%s' % (FLAGS.dataset, epoch),
                  dir, sess, feed_dict)
        print_var(model.loss,
                  'nane__%s_loss_%s' % (FLAGS.dataset, epoch), dir, sess,
                  feed_dict)

    # Training step
    fetches = [model.opt_op, model.loss]
    preds = model.outputs
    outs = sess.run(fetches, feed_dict=feed_dict)
    loss = outs[1]

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=",
          "{:.5f}".format(loss),
          "time=",
          "{:.5f}".format(time.time() - t))

print("done")