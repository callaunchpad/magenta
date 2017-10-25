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
import tensorflow as tf

from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops.rnn_cell_impl import RNNCell
from math import pi, floor

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

class PhaseFunctionedLSTM(RNNCell):

	def __init__(self, rng=rng, input_shape, output_shape, dropout=0.5):


       	return

    def __call__(self, input):


        return

    def cost(self, input):

        return

    def save(self, database, prefix=''):


        return

    def load(self, database, prefix=''):

    	return
