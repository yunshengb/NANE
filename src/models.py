from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None



    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """

        def call_layer(layer, stop=False):
            hidden = layer(self.activations[-1])
            if stop:
                hidden = tf.stop_gradient(hidden)
            self.activations.append(hidden)

        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            call_layer(layer)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError


class NANE(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(NANE, self).__init__(**kwargs)

        self.inputs = placeholders.get('features')
        self.input_dim = input_dim
        if FLAGS.embed == 2 or FLAGS.embed == 3:
            self.usl_output_dim = \
            placeholders['usl_labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Cross entropy error
        self.usl_labels = self.placeholders['usl_labels']
        loss = masked_softmax_cross_entropy(self.outputs,
                                            self.usl_labels,
                                            None,
                                            model=self)
        self.loss += loss

    def _accuracy(self):
        self.accuracy = accuracy(self.outputs,
                                 self.placeholders['labels'])

    def _build(self):



        if FLAGS.embed == 2 or FLAGS.embed == 3:
            self.layers.append(NeighborAggregation(input_dim=self.input_dim,
                                                output_dim=FLAGS.hidden1,
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=0,
                                                sparse_inputs=False,
                                                featureless=self.inputs is None,
                                                logging=self.logging))

            self.layers.append(NeighborAggregation(input_dim=FLAGS.hidden1,
                                                output_dim=FLAGS.hidden2,
                                                placeholders=self.placeholders,
                                                act=lambda x : x,
                                                dropout=0,
                                                sparse_inputs=False,
                                                logging=self.logging))

        if FLAGS.embed == 2 or FLAGS.embed == 3:
            if FLAGS.embed == 2:
                layers = self.layers
            else:
                self.usl_layers = []
                layers = self.usl_layers
            layers.append(Embedding(input_dim=100,
                                    output_dim=self.usl_output_dim,
                                    placeholders=self.placeholders,
                                    act=lambda x: x,
                                    dropout=False,
                                    logging=self.logging,
                                    model=self))

    def predict(self):
        return tf.nn.softmax(self.outputs)
