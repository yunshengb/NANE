from inits import *
import tensorflow as tf
from neg_sampling import neg_sampling
from math import ceil

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            self.outputs = outputs
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""

    def __init__(self, input_dim, output_dim, placeholders,
                 sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')


    def _call(self, inputs):
        x = inputs

        # transform
        if not self.featureless:
            output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)
        else:
            output = self.vars['weights']

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class NeighborAggregation(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(NeighborAggregation, self).__init__(**kwargs)


        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(
                                                            i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')


    def _call(self, inputs):
        x = inputs

        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class Embedding(Layer):
    """Graph embedding layer."""

    def __init__(self, input_dim, output_dim, placeholders,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, model=None, **kwargs):
        super(Embedding, self).__init__(**kwargs)


        self.act = act
        self.sparse_inputs = sparse_inputs

        if not FLAGS.need_batch:
            self.sims_mask = placeholders['sims_mask']
        else:
            self.batch = placeholders['batch']
            self.labels = placeholders['usl_labels']
            self.pos_labels = placeholders['pos_labels']
            self.neg_labels = placeholders['neg_labels']
            self.num_data = placeholders['num_data']


        self.model = None
        if model:
            self.model = model

    def _call(self, inputs):
        x = inputs

        self.embeddings = x

        if hasattr(self, 'batch'):
            self.embeddings = x
            print('num_data', self.num_data)
            output = neg_sampling(self.embeddings, self.batch, self.pos_labels,
                                  self.neg_labels)
        else:
            embed = self.embeddings

            output = tf.matmul(embed, tf.transpose(self.embeddings))
            output = tf.multiply(output, self.sims_mask)
        self.output = output

        return output
