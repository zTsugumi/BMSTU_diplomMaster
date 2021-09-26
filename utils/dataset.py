import tensorflow as tf
import tensorflow_datasets as tfds
import json
from utils import preprocess_mnist, preprocess_smallnorb, preprocess_cifar10


class Dataset(object):
    '''
    This constructs functions to process dataset
    '''

    def __init__(self, data_name, conf_path='conf.json', custom_dir=''):
        self.data_name = data_name
        self.conf_path = conf_path
        self.load_config()
        self.load_dataset(custom_dir)

    def load_config(self):
        with open(self.conf_path) as f_conf:
            self.conf = json.load(f_conf)

    def load_dataset(self, custom_dir):
        try:
            if self.data_name == 'MNIST':
                if not custom_dir:
                    (self.x_train, self.y_train), (self.x_test_orig, self.y_test) = \
                        tf.keras.datasets.mnist.load_data()
                    self.x_train, self.y_train = preprocess_mnist.pre_process(
                        self.x_train, self.y_train)
                    self.x_test, self.y_test = preprocess_mnist.pre_process(
                        self.x_test_orig, self.y_test)
                else:
                    img = tf.io.read_file(custom_dir)
                    img = tf.io.decode_image(img, channels=1, dtype=tf.float32)
                    img = tf.image.resize(img, [28, 28])
                    self.x_custom = tf.reshape(img, shape=[1, 28, 28, 1])

                self.class_names = list(range(10))

            elif self.data_name == 'SMALLNORB':
                if not custom_dir:
                    (data_train, data_test), data_info = tfds.load(
                        name='smallnorb',
                        split=['train', 'test'],
                        data_dir=self.conf['dir_data'],
                        shuffle_files=True,
                        as_supervised=False,
                        with_info=True
                    )
                    self.x_train, self.y_train = preprocess_smallnorb.pre_process(
                        data_train)
                    self.x_test, self.y_test, self.x_test_orig = preprocess_smallnorb.pre_process(
                        data_test)
                    self.x_test, self.y_test = preprocess_smallnorb.pre_process_test(
                        self.x_test, self.y_test)
                    self.class_names = data_info.features['label_category'].names
                else:
                    img = tf.io.read_file(custom_dir)
                    img = tf.io.decode_image(
                        img, channels=1, dtype=tf.float32)
                    img = tf.image.resize(img, [48, 48])
                    bound = 8
                    img = img[bound:-bound, bound:-bound, :]
                    img = tf.tile(img, [1, 1, 2])
                    self.x_custom = tf.reshape(img, shape=[1, 32, 32, 2])
                    self.class_names = [
                        'animal', 'human', 'plane', 'truck', 'car']

            elif self.data_name == 'CIFAR10':
                if not custom_dir:
                    (self.x_train, self.y_train), (self.x_test_orig, self.y_test) = \
                        tf.keras.datasets.cifar10.load_data()
                    self.x_train, self.y_train = preprocess_cifar10.pre_process(
                        self.x_train, self.y_train)
                    self.x_test, self.y_test = preprocess_cifar10.pre_process(
                        self.x_test_orig, self.y_test)
                else:
                    img = tf.io.read_file(custom_dir)
                    img = tf.io.decode_image(
                        img, channels=3, dtype=tf.float32)
                    # img.set_shape([32, 32, 3])
                    img = tf.image.resize(img, [32, 32])
                    img = tf.image.per_image_standardization(img)
                    self.x_custom = tf.reshape(img, shape=[1, 32, 32, 3])

                self.class_names = ['airplane', 'automobile', 'bird',
                                    'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            else:
                raise RuntimeError(
                    f'data_name {self.data_name} not recognized')
        except:
            print('Something')
            pass

    def get_tf_data(self):
        if self.data_name == 'MNIST':
            data_train, data_test = preprocess_mnist.generate_tf_data(
                self.x_train, self.y_train,
                self.x_test, self.y_test,
                self.conf['batch_size']
            )
        elif self.data_name == 'SMALLNORB':
            data_train, data_test = preprocess_smallnorb.generate_tf_data(
                self.x_train, self.y_train,
                self.x_test, self.y_test,
                self.conf['batch_size']
            )
        elif self.data_name == 'CIFAR10':
            data_train, data_test = preprocess_cifar10.generate_tf_data(
                self.x_train, self.y_train,
                self.x_test, self.y_test,
                self.conf['batch_size']
            )
        else:
            raise RuntimeError(f'data_name {self.data_name} not recognized')

        return data_train, data_test
