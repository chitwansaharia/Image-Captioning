import sys
sys.path.insert(1,"/u/sahariac/.local/lib/python2.7/site-packages/")
import tensorflow as tf
import collections

_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class LSTMStateTuple(_LSTMStateTuple):
    """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

    Stores two elements: `(c, h)`, in that order.

    Only used when `state_is_tuple=True`.
    """
    __slots__ = ()

    @property
    def dtype(self):
        (c, h) = self
        if not c.dtype == h.dtype:
            raise TypeError("Inconsistent internal state: %s tf %s" %
                            (str(c.dtype), str(h.dtype)))
        return c.dtype


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable(
            "Matrix", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size],
            dtype=dtype,
            initializer=tf.constant_initializer(
                bias_start, dtype=dtype))
    return res + bias_term


class myBasicLSTMCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_units, forget_bias=1.0, input_size=None,
                 state_is_tuple=True, activation=tf.nn.tanh):
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(axis=1, num_or_size_splits=2, value=state)
            concat = _linear(inputs + [h], 1 * self._num_units, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            #i, j, f, o = tf.split(1, 4, concat)

            # new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
            #         self._activation(j))
            #new_h = self._activation(new_c) * tf.sigmoid(o)

            new_c = new_h = self._activation(concat)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = array_ops.concat(1, [new_c, new_h])
            return new_h, new_state


class myLSTMCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_units, forget_bias=1.0, input_size=None,
                 state_is_tuple=True, use_peepholes=True, activation=tf.nn.tanh):
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._use_peepholes = use_peepholes

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(axis=1, num_or_size_splits=2, value=state)
            concat = _linear(inputs + [h], 4 * self._num_units, True)

            #i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(axis=1, num_or_size_splits=4, value=concat)

            # Diagonal connections
            if self._use_peepholes:
                dtype = inputs[0].dtype
                w_f_diag = tf.get_variable(
                    "W_F_diag", shape=[self._num_units], dtype=dtype)
                w_i_diag = tf.get_variable(
                    "W_I_diag", shape=[self._num_units], dtype=dtype)
                w_o_diag = tf.get_variable(
                    "W_O_diag", shape=[self._num_units], dtype=dtype)

            if self._use_peepholes:
                new_c = (tf.sigmoid(f + self._forget_bias + w_f_diag * c) * c +
                         tf.sigmoid(i + w_i_diag * c) * self._activation(j))
                new_h = self._activation(new_c) * tf.sigmoid(o + w_o_diag * new_c)
            else:
                new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
                         self._activation(j))
                new_h = self._activation(new_c) * tf.sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = array_ops.concat(1, [new_c, new_h])
            return new_h, new_state
