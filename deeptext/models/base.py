import os
import tensorflow as tf

import deeptext.models.constants as constants


class Base(object):

    def __init__(self, params):
        assert constants.PARAM_KEY_MODEL_DIR in params, ("invalid parameter, %s is missing" % constants.PARAM_KEY_MODEL_DIR)
        assert constants.PARAM_KEY_MODEL_NAME in params, ("invalid parameter, %s is missing" % constants.PARAM_KEY_MODEL_NAME)

        self.params = params

        tf.logging.set_verbosity(tf.logging.INFO)

    def fit(self, steps, batch_size, training_data_path, validation_data_path=None):
        raise NotImplementedError()

    def save(self, sess):
        saver = tf.train.Saver()
        filepath = self.params[constants.PARAM_KEY_MODEL_DIR] + '/' + self.params[constants.PARAM_KEY_MODEL_NAME]
        saver.save(sess, filepath)

    def ready_to_restore(self):
        filepath = self.params[constants.PARAM_KEY_MODEL_DIR] + '/' + self.params[constants.PARAM_KEY_MODEL_NAME]
        filepath_meta = filepath + '.meta'

        return os.path.isfile(filepath_meta)

    def restore(self, sess):
        filepath = self.params[constants.PARAM_KEY_MODEL_DIR] + '/' + self.params[constants.PARAM_KEY_MODEL_NAME]
        filepath_meta = filepath + '.meta'

        if not os.path.isfile(filepath_meta):
            return False
        #  saver = tf.train.import_meta_graph(filepath_meta)
        saver = tf.train.Saver()
        saver.restore(sess, filepath)
        return True
