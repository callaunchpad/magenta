"""
#=============================#
||                           ||
||                           ||
||        PFNN Class         ||
||                           ||
||                           ||
#=============================#

Classes for Phase Functioned Neural Networks.
We will be implementing these from scratch.

We will be basing these off of code in Theano
from The Orange Duck found here:
https://github.com/sreyafrancis/PFNN


"""

import numpy as np
import collections
import tensorflow as tf

from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops.rnn_cell_impl import RNNCell
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops
from math import pi, floor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.rnn import AttentionCellWrapper
from tensorflow.contrib.rnn import DropoutWrapper

sigmoid = math_ops.sigmoid
tanh = math_ops.tanh

class PhaseFunctionedFFNN(base_layer.Layer):

    # only one layer for demonstration purposes
    def __init__(self, input_shape, output_shape, dropout=0.5):
        self.phases = 4
        self.x = tf.placeholder(tf.float32, [None, input_shape])
        self.W0 = [tf.Variable(tf.zeros([input_shape, output_shape])) for _ in range(self.phases)]
        self.b0 = [tf.Variable(tf.zeros([output_shape])) for _ in range(self.phases)]
        self.layers = [self.W0, self.b0]

        return

    def __call__(self, input):
        phase = input[-1]
        input = input[:-1]
        phase_num = (4 * phase) / (2 * pi)

        phase_depth = phase_num % 1 # how far into the current phase we are
        k = lambda n: (floor(phase_num) + n - 1) % 4
        W0_phase = self.cubic_spline(self.W0[k(0)], self.W0[k(1)], self.W0[k(2)], self.W0[k(3)], w)
        b0_phase = self.cubi|c_spline(self.b0[k(0)], self.b0[k(1)], self.b0[k(2)], self.b0[k(3)], w)

        return tf.matmul(W0_phase, input) + b0_phase

    def cubic_spline(self, y0, y1, y2, y3, mu):
        return ( \
            (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + \
            (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + \
            (-0.5*y0+0.5*y2)*mu + \
            (y1))



PFLSTMStateTuple = collections.namedtuple("PFLSTMStateTuple", ("c", "h"))

class PhaseFunctionedLSTM(RNNCell):

    def __init__(self, input_shape, output_shape, dropout=0.5):
        self.phases = 4

        self.forget_gate = [tf.Variable(tf.zeros([input_shape, output_shape]))] * self.phases
        self.forget_bias = [tf.Variable(tf.zeros([output_shape]))] * self.phases

        self.input_gate = [tf.Variable(tf.zeros([input_shape, output_shape]))] * self.phases
        self.input_bias = [tf.Variable(tf.zeros([output_shape]))] * self.phases

        self.new_input = [tf.Variable(tf.zeros([input_shape, output_shape]))] * self.phases
        self.new_bias = [tf.Variable(tf.zeros([output_shape]))] * self.phases

        self.output_gate = [tf.Variable(tf.zeros([input_shape, output_shape]))] * self.phases
        self.output_bias = [tf.Variable(tf.zeros([output_shape]))] * self.phases

        self.layers = [self.forget_gate, self.forget_bias, self.input_gate, self.input_bias, \
                        self.new_input, self.new_bias, self.output_gate, self.output_bias]
        return

    def __call__(self, input, state):
                # (c, h) = state
                # input = x
        # right now assumes only one input at a time (i.e. input is just a vector)
        h = state[1]
        x = input
        phase = input[-1]
        input = input[:-1]
        phase_num = (4 * phase) / (2 * pi) # assumes phase is from 0 - 2pi

        phase_depth = phase_num % 1 # how far into the current phase we are
        k = lambda n: (floor(phase_num) + n - 1) % 4 # control point selector function

        # indices 0-1 = forget, 2-3 = input, 4-5 = new, 6-7 = output
        phased_layers = []
        for layer in self.layers:
            interpolated = self.cubic_spline(self.layer[k(0)], self.layer[k(1)], self.layer[k(2)], self.layer[k(3)], w)
            phased_layers.append(interpolated) # W values

            concat = tf.concat([h, x], 1)
            W_f = phased_layers[0] # forget Weights
            b_f = phased_layers[1] # forget bias
            W_i = phased_layers[2] # input Weights
            b_i = phased_layers[3] # input bias
            W_c = phased_layers[4] # new input weights
            b_c = phased_layers[5] # new input bias
            W_o = phased_layers[6] # output weights
            b_o = phased_layers[7] # output bias
            f = sigmoid(tf.matmul(W_f, concat) + b_f)
            i = sigmoid(tf.matmul(W_i, concat) + b_i)
            C_tilde = tanh(tf.matmul(W_c, concat) + b_c)
            o = sigmoid(tf.matmul(W_o, concat + b_o))
            new_c = f * c + i * C_tilde
            new_h = o * tanh(new_c)
            new_state = PFLSTMStateTuple(new_c, new_h)

        return (new_h, new_state)


    def cubic_spline(self, y0, y1, y2, y3, mu):
        return ( \
            (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + \
            (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + \
            (-0.5*y0+0.5*y2)*mu + \
            (y1))


class PhaseAttentionCellWrapper(AttentionCellWrapper):
    """Changing basic attention cell wrapper to incorporate phase.
    Implementation based on https://arxiv.org/abs/1409.0473.
    """

    def __init__(self, cell, attn_length, attn_size=None, attn_vec_size=None,
               input_size=None, state_is_tuple=True, reuse=None):
    """Create a cell with attention.
    Args:
      cell: an RNNCell, an attention is added to it.
      attn_length: integer, the size of an attention window.
      attn_size: integer, the size of an attention vector. Equal to
          cell.output_size by default.
      attn_vec_size: integer, the number of convolutional features calculated
          on attention state and a size of the hidden layer built from
          base cell state. Equal attn_size to by default.
      input_size: integer, the size of a hidden linear layer,
          built from inputs and attention. Derived from the input tensor
          by default.
      state_is_tuple: If True, accepted and returned states are n-tuples, where
        `n = len(cells)`.  By default (False), the states are all
        concatenated along the column axis.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if cell returns a state tuple but the flag
          `state_is_tuple` is `False` or if attn_length is zero or less.
    """
    super(AttentionCellWrapper, self).__init__(_reuse=reuse)
    if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
        raise TypeError("The parameter cell is not RNNCell.")
    if nest.is_sequence(cell.state_size) and not state_is_tuple:
        raise ValueError("Cell returns tuple of states, but the flag "
                       "state_is_tuple is not set. State size is: %s"
                       % str(cell.state_size))
    if attn_length <= 0:
        raise ValueError("attn_length should be greater than zero, got %s"
                       % str(attn_length))
    if not state_is_tuple:
        logging.warn(
          "%s: Using a concatenated state is slower and will soon be "
          "deprecated.  Use state_is_tuple=True.", self)
    if attn_size is None:
        attn_size = cell.output_size
    if attn_vec_size is None:
        attn_vec_size = attn_size
    self._state_is_tuple = state_is_tuple
    self._cell = cell
    self._attn_vec_size = attn_vec_size
    self._input_size = input_size -1  # discount phase
    self._attn_size = attn_size
    self._attn_length = attn_length
    self._reuse = reuse
    self._linear1 = None
    self._linear2 = None
    self._linear3 = None
    self.phase = None

    def call(self, inputs, state):
    """Long short-term memory cell with attention (LSTMA)."""

    # store phase, shorten inputs
    self.phase = inputs[-1]
    inputs = inputs[:-1]

    if self._state_is_tuple:
        state, attns, attn_states = state
    else:
        states = state
        state = array_ops.slice(states, [0, 0], [-1, self._cell.state_size])
        attns = array_ops.slice(
          states, [0, self._cell.state_size], [-1, self._attn_size])
        attn_states = array_ops.slice(
          states, [0, self._cell.state_size + self._attn_size],
          [-1, self._attn_size * self._attn_length])
    attn_states = array_ops.reshape(attn_states,
                                    [-1, self._attn_length, self._attn_size])
    input_size = self._input_size
    if input_size is None:
        input_size = inputs.get_shape().as_list()[1]
    if self._linear1 is None:
        self._linear1 = _Linear([inputs, attns], input_size, True)

    inputs = self._linear1([inputs, attns])

    # append phase back into input so that PFNN can use it
    inputs.append(self.phase)

    cell_output, new_state = self._cell(inputs, state)
    if self._state_is_tuple:
        new_state_cat = array_ops.concat(nest.flatten(new_state), 1)
    else:
        new_state_cat = new_state
    new_attns, new_attn_states = self._attention(new_state_cat, attn_states)
    with vs.variable_scope("attn_output_projection"):
        if self._linear2 is None:
            self._linear2 = _Linear([cell_output, new_attns], self._attn_size, True)
        output = self._linear2([cell_output, new_attns])
    new_attn_states = array_ops.concat(
        [new_attn_states, array_ops.expand_dims(output, 1)], 1)
    new_attn_states = array_ops.reshape(
        new_attn_states, [-1, self._attn_length * self._attn_size])
    new_state = (new_state, new_attns, new_attn_states)
    if not self._state_is_tuple:
        new_state = array_ops.concat(list(new_state), 1)

    return output, new_state

class PhaseDropoutWrapper(DropoutWrapper):
  """Operator adding dropout to inputs and outputs of the given cell.
     Incorporates phase. """

    def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
    state_keep_prob=1.0, variational_recurrent=False,
    input_size=None, dtype=None, seed=None,
    dropout_state_filter_visitor=None):

    super().__init__(self, cell, input_keep_prob, output_keep_prob, state_keep_prob, variational_recurrent,
        input_size, dtype, seed, dropout_state_filter_visitor)

    #don't know if I need this
    if input_size:
        self.input_size = input_size -1

    self.phase = None
    return

    def __call__(self, inputs, state, scope=None):
    """Run the cell with the declared dropouts."""

    # store phase value
    self.phase = inputs[-1]
    inputs = inputs[:-1]

    def _should_dropout(p):
         return (not isinstance(p, float)) or p < 1

    if _should_dropout(self._input_keep_prob):
        inputs = self._dropout(inputs, "input",
                             self._recurrent_input_noise,
                             self._input_keep_prob)

    # re-append phase so PFNN can use it
    inputs.append(self.phase)

    output, new_state = self._cell(inputs, state, scope)

    if _should_dropout(self._state_keep_prob):
    #       Identify which subsets of the state to perform dropout on and
      # which ones to keep.
        shallow_filtered_substructure = nest.get_traverse_shallow_structure(
                                self._dropout_state_filter, new_state)
        new_state = self._dropout(new_state, "state",
                                self._recurrent_state_noise,
                                self._state_keep_prob,
                                shallow_filtered_substructure)
    if _should_dropout(self._output_keep_prob):
        output = self._dropout(output, "output",
                             self._recurrent_output_noise,
                             self._output_keep_prob)
    return output, new_state
