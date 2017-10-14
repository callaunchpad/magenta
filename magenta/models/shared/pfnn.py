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

class PhaseFunctionedFFNN(base_layer.Layer):

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


