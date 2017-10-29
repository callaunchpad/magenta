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

sigmoid = math_ops.sigmoid
tanh = math_ops.tanh

class PhaseFunctionedFFNN(base_layer.Layer):

	# only one layer for demonstration purposes
	def __init__(self, rng=rng, input_shape, output_shape, dropout=0.5):
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
        b0_phase = self.cubic_spline(self.b0[k(0)], self.b0[k(1)], self.b0[k(2)], self.b0[k(3)], w)

        return tf.matmul(W0_phase, input) + b0_phase

    def cost(self, input):

        return

    def save(self, database, prefix=''):

        return

    def load(self, database, prefix=''):

    	return

	def cubic_spline(self, y0, y1, y2, y3, mu):
	    return ( \
	        (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + \
	        (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + \
	        (-0.5*y0+0.5*y2)*mu + \
	        (y1))



PFLSTMStateTuple = collections.namedtuple("PFLSTMStateTuple", ("c", "h"))

class PhaseFunctionedLSTM(RNNCell):

	def __init__(self, rng=rng, input_shape, output_shape, dropout=0.5):
		self.phases = 4

		self.forget_gate = [tf.Variable(tf.zeros([input_shape, output_shape]))] * self.phases
		self.forget_bias = [tf.Variable(tf.zeros([output_shape]))] * self.phases

		self.input_gate = [tf.Variable(tf.zeros([input_shape, output_shape]))] * self.phases
		self.input_bias = [tf.Variable(tf.zeros([output_shape]))] * self.phases

		self.new_input = [tf.Variable(tf.zeros([input_shape, output_shape]))] * self.phases
		self.new_bias = [tf.Variable(tf.zeros([output_shape]))] * self.phases

		self.output_gate = [tf.Variable(tf.zeros([input_shape, output_shape]))] * self.phases
		self.output_bias = [tf.Variable(tf.zeros([output_shape]))] * self.phases

		self.layers = [self.forget_gate, self.forget_bias, self.input_gate, self.input_bias\
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
                o = sigmoid(tf.matmul(W_o, concat + b_o)
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

    def cost(self, input):

        return

    def save(self, database, prefix=''):


        return

    def load(self, database, prefix=''):

    	return
