import tensorflow as tf

from tensorflow.python.eager import context

class PFNNSaver(tf.train.Saver):

    def build(self):
	    if context.in_eager_mode():
	      raise RuntimeError("Use save/restore instead of build in eager mode.")
	    self._build(self._filename, build_save=True, build_restore=False)
