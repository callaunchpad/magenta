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
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.rnn import AttentionCellWrapper
from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.python.ops.rnn_cell_impl import _linear
import tensorflow.python.ops.rnn_cell_impl
from tensorflow.python.estimator import util as estimator_util
from tensorflow.python.layers.base import Layer
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

sigmoid = math_ops.sigmoid
tanh = math_ops.tanh
_WEIGHTS_VARIABLE_NAME = "kernel"
_BIAS_VARIABLE_NAME = "bias"
_Linear = rnn_cell_impl._Linear

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
        if(len(input.shape)>1):
            phase = input[:,-1]
            input = input[:,:-1]
        else:
            phase = input[-1]
            input = input[:-1]
        phase_num = (4 * phase) 

        phase_depth = phase_num % 1 # how far into the current phase we are
        k = lambda n: ((phase_num)//1 + n - 1) % 4
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

class PhaseFunctionedLSTM(BasicLSTMCell):


    def __init__(self, num_units, forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None):
        Layer.__init__(self)
        self.phases = 4
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh
        self._dtype = tf.float32
        return

    def __call__(self, inputs, state):
                # (c, h) = state
                # input = x
        # right now assumes only one input at a time (i.e. input is just a vector)
        print("\ncalled\n")
        if self._state_is_tuple:
          c, h = state
        else:
          c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

        if(len(inputs.shape)>1):
            inputs, phase = array_ops.split(inputs, [inputs.shape[1].value - 1, 1], axis=1)
            # phase = inputs[:,-1]
            # inputs = inputs[:,:-1]
        else:
            inputs, phase = array_ops.split(inputs, [inputs.shape[0].value - 1, 1], axis=0)
            # phase = inputs[-1]
            # inputs = inputs[:-1]

        if (not self.built):
            self.build(inputs.shape)

        phase_num = (4 * phase) # assumes phase is from 0 - 1

        phase_depth = phase_num % 1 # how far into the current phase we are
        k = lambda n: (phase_num//1 + n - 1) % 4 # control point selector function
        w = phase_depth

        phased_temp = [None for _ in range(self.phases)]
        kernel_split = array_ops.split(self._kernel, self.phases, axis=1)
        bias_split = array_ops.split(self._bias, self.phases, axis=0)

        for i in range(self.phases):
            concat = math_ops.matmul(
                array_ops.concat([inputs, h], 1), kernel_split[i])
            concat = nn_ops.bias_add(concat, bias_split[i])
            phased_temp[i] = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

        i = self.cubic_spline(phased_temp[0][0], phased_temp[1][0], phased_temp[2][0], phased_temp[3][0], w)
        j = self.cubic_spline(phased_temp[0][1], phased_temp[1][1], phased_temp[2][1], phased_temp[3][1], w)
        f = self.cubic_spline(phased_temp[0][2], phased_temp[1][2], phased_temp[2][2], phased_temp[3][2], w)
        o = self.cubic_spline(phased_temp[0][3], phased_temp[1][3], phased_temp[2][3], phased_temp[3][3], w)

        new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))

        # calculate the output by running activation on the cell state and multiplying
        # with a filtered version of the [input, h]
        new_h = self._activation(new_c) * sigmoid(o)

        if self._state_is_tuple:
          new_state = PFLSTMStateTuple(new_c, new_h)
        else:
          new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state


    def cubic_spline(self, y0, y1, y2, y3, mu):
        return ( \
            (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + \
            (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + \
            (-0.5*y0+0.5*y2)*mu + \
            (y1))

    def build(self, inputs_shape):
        if inputs_shape[1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % inputs_shape)

        print("\n------------------------")
        print(type(inputs_shape[1]))
        print(inputs_shape[1])
        print("------------------------\n")

        input_depth = inputs_shape[1].value
        h_depth = self._num_units
        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + h_depth, 4 * self._num_units * self.phases])
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[4 * self._num_units * self.phases],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    @property
    def state_size(self):
        return (PFLSTMStateTuple(self._num_units, self._num_units)
                    if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units


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
        if input_size:
            self._input_size = input_size -1  # discount phase
        else:
            self._input_size = input_size
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

        if(len(inputs.shape)>1):
            inputs, self.phase = array_ops.split(inputs, [inputs.shape[1].value - 2, 2], axis=1)
            reap = 1
            # print(phase)
            # phase = inputs[:,-1]
            # inputs = inputs[:,:-1]
        else:
            inputs, self.phase = array_ops.split(inputs, [inputs.shape[0].value - 1, 1], axis=0)
            reap = 0
            # print(phase)
            # phase = inputs[-1]
            # inputs = inputs[:-1]


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
        # inputs.append(self.phase)
        inputs = array_ops.concat([inputs, self.phase], reap)

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
        
        # super(PhaseDropoutWrapper, self).__init__(cell, input_keep_prob, output_keep_prob, state_keep_prob, variational_recurrent,
        #     input_size, dtype, seed, dropout_state_filter_visitor)
        super(PhaseDropoutWrapper, self).__init__(cell, input_keep_prob, output_keep_prob=output_keep_prob, state_keep_prob=state_keep_prob, 
            variational_recurrent=variational_recurrent, input_size=input_size, 
            dtype=dtype, seed=seed)
        #don't know if I need this
        if input_size:
            self.input_size = input_size -1

        self.phase = None
        return

    def __call__(self, inputs, state, scope=None):
        """Run the cell with the declared dropouts."""

        # store phase value

        if(len(inputs.shape)>1):
            inputs, self.phase = array_ops.split(inputs, [inputs.shape[1].value -1, 1], axis=1)
            reap = 1
            print(inputs.shape)
            print(self.phase.shape)
            # print(phase)
            # phase = inputs[:,-1]
            # inputs = inputs[:,:-1]
        else:
            inputs, self.phase = array_ops.split(inputs, [inputs.shape[0].value - 1, 1], axis=0)
            reap = 0
            # phase = inputs[-1]
            # inputs = inputs[:-1]

        def _should_dropout(p):
             return (not isinstance(p, float)) or p < 1

        if _should_dropout(self._input_keep_prob):
            inputs = self._dropout(inputs, "input",
                                 self._recurrent_input_noise,
                                 self._input_keep_prob)

        # re-append phase so PFNN can use it

        inputs = array_ops.concat([inputs, self.phase], reap)

        output, new_state = self._cell(inputs, state)

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


