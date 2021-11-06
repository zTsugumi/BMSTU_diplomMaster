import json
from utils import download
from utils.cifar10 import preprocess_cifar10, generate_tf_cifar10
from utils.cifar10.cifar_dataset import load_cifar10
from utils.lfw import load_lfw, preprocess_lfw, generate_tf_lfw


class Dataset(object):
  '''
  This constructs functions to process dataset
  '''

  def __init__(self, data_name, conf_path='conf.json', custom_dir='', dl=False):
    self.data_name = data_name
    self.conf_path = conf_path
    self.load_config()
    self.load_dataset(custom_dir, dl)

  def load_config(self):
    with open(self.conf_path) as f_conf:
      self.conf = json.load(f_conf)

  def load_dataset(self, custom_dir, dl=False):
    try:
      if self.data_name == 'LFW':
        dir = custom_dir if custom_dir else './data/lfw'

        if dl:
          download.download_lfw(dir)

        self.data_orig, self.attrs = load_lfw(dir, use_raw=True, dimx=32, dimy=32)
        self.data = preprocess_lfw(self.data_orig)

      elif self.data_name == 'CIFAR10':
        dir = custom_dir if custom_dir else './data/cifar10'

        if dl:
          download.download_cifar10(dir)

        (self.x_train, self.y_train), (self.x_val, self.y_val), (self.x_test_orig, self.y_test) = \
            load_cifar10(dir)

        self.x_train, self.y_train = preprocess_cifar10(
            self.x_train, self.y_train)
        self.x_val, self.y_val = preprocess_cifar10(
            self.x_val, self.y_val)
        self.x_test, self.y_test = preprocess_cifar10(
            self.x_test_orig, self.y_test)

        self.class_names = ['airplane', 'automobile', 'bird',
                            'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

      else:
        raise RuntimeError(
            f'data_name {self.data_name} not recognized')
    except e:
      print('Error loading dataset', e)
      pass

  def get_tf_data(self):
    if self.data_name == 'LFW':
      data_train, data_test = generate_tf_lfw(
          self.X,
          self.conf['batch_size']
      )
    elif self.data_name == 'CIFAR10':
      data_train, data_test = preprocess_cifar10.generate_tf_cifar10(
          self.x_train, self.y_train,
          self.x_test, self.y_test,
          self.conf['batch_size']
      )
    else:
      raise RuntimeError(f'data_name {self.data_name} not recognized')

    return data_train, data_test
