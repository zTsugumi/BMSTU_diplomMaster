import numpy as np
import tensorflow as tf
import json


class Model(object):
    '''
    This contructs the abstract class for all type of models
    '''

    def __init__(self, name, mode, conf_path='conf.json'):
        self.name = name
        self.mode = mode
        self.conf_path = conf_path
        self.model = None
        self.load_conf()

    def load_conf(self):
        with open(self.conf_path) as f_conf:
            self.conf = json.load(f_conf)

    def load_weight(self, epoch, file_name=""):
        if not file_name:
            file_name = f'/weights-{epoch:02d}.h5'
            dir_model = self.dir_model + file_name
        else:
            dir_model = file_name

        try:
            self.model.load_weights(dir_model)
            print(f'Load {file_name} successful')
            return True
        except:
            print(f'[WARNING] {dir_model} not found')
            return False

    def save_weight(self, epoch):
        file_name = f'/weights-{epoch:02d}.h5'
        dir_model = self.dir_model + file_name
        try:
            self.model.save_weights(dir_model)
            print(f'Save {file_name} successful')
        except:
            print(f'[WARNING] {dir_model} not found')

    def predict(self, x):
        y_pred, x_reconstruct = self.model.predict(x)
        label = tf.argmax(y_pred, axis=1)
        acc = tf.reduce_max(y_pred, axis=1)

        return y_pred, label, acc, x_reconstruct

    def evaluate(self, x_test, y_test):
        y_pred, _ = self.model.predict(x_test)
        correct = tf.reduce_sum(
            tf.cast(tf.argmax(y_pred, axis=1) == tf.argmax(y_test, axis=1), tf.float32))
        accuracy = correct / y_test.shape[0]
        test_err = 1.0 - accuracy

        return accuracy, test_err
