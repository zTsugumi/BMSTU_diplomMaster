import json
from . import download
from .cifar10 import load_cifar10, preprocess_cifar10, generate_tf_cifar10
from .lfw import load_lfw, preprocess_lfw, generate_tf_lfw
from .geo import build_image, load_geo, generate_tf_geo


class Dataset(object):
  '''
  This constructs functions to process dataset
  '''

  def __init__(self, data_name, conf_path='conf.json'):
    self.data_name = data_name
    self.conf_path = conf_path
    self.load_config()

  def load_config(self):
    with open(self.conf_path) as f_conf:
      self.conf = json.load(f_conf)


class DatasetCIFAR10(Dataset):
  def __init__(self, data_name, conf_path='conf.json', custom_dir='',
               dl=False, val_percent=0.1):
    super(DatasetCIFAR10, self).__init__(data_name, conf_path)
    self.load_dataset(custom_dir, dl, val_percent)

  def load_dataset(self, custom_dir, dl, val_percent):
    try:
      if self.data_name == 'CIFAR10':
        dir = custom_dir if custom_dir else './data/cifar10'

        if dl:
          download.download_cifar10(dir)

        (self.x_train, self.y_train),
        (self.x_val, self.y_val),
        (self.x_test_orig, self.y_test) =\
            load_cifar10(dir, val_percent)

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
    except:
      print('Error loading dataset')
      pass

  def get_tf_data(self):
    try:
      if self.data_name == 'CIFAR10':
        data_train, data_val, data_test = generate_tf_cifar10(
            self.x_train, self.y_train,
            self.x_val, self.y_val,
            self.x_test, self.y_test,
            self.conf['batch_size']
        )
      else:
        raise RuntimeError(f'data_name {self.data_name} not recognized')
    except:
      print('Error loading dataset')
      pass

    return data_train, data_val, data_test


class DatasetLFW(Dataset):  # WIP
  def __init__(self, data_name, conf_path='conf.json', custom_dir='', dl=False):
    super(DatasetLFW, self).__init__(data_name, conf_path)
    self.load_dataset(custom_dir, dl)

  def load_dataset(self, custom_dir, dl):
    try:
      if self.data_name == 'LFW':
        dir = custom_dir if custom_dir else './data/lfw'

        if dl:
          download.download_lfw(dir)

        self.data_orig, self.attrs = load_lfw(dir, use_raw=True, dimx=32, dimy=32)
        self.data = preprocess_lfw(self.data_orig)
      else:
        raise RuntimeError(
            f'data_name {self.data_name} not recognized')
    except:
      print('Error loading dataset')
      pass

  def get_tf_data(self):
    try:
      if self.data_name == 'LFW':
        data_train, data_test = generate_tf_lfw(
            self.data,
            self.conf['batch_size']
        )
      else:
        raise RuntimeError(f'data_name {self.data_name} not recognized')
    except:
      print('Error loading dataset')
      pass

    return data_train, data_test


class DatasetVOXCELEB(Dataset):  # WIP
  def __init__(self, data_name, conf_path='conf.json', custom_dir='', dl=False):
    super(DatasetVOXCELEB, self).__init__(data_name, conf_path)
    self.load_dataset(custom_dir, dl)

  def load_dataset(self, custom_dir, dl):
    dir = custom_dir if custom_dir else './data/voxceleb'

    if dl:
      download.download_voxceleb(dir)


class DatasetGEO(Dataset):
  def __init__(self, data_name, conf_path='conf.json', custom_dir='',
               gen=False, val_percent=0.1, test_percent=0.1):
    super(DatasetGEO, self).__init__(data_name, conf_path)
    self.load_dataset(custom_dir, gen, val_percent, test_percent)

  def load_dataset(self, custom_dir, gen, val_percent, test_percent):
    try:
      if self.data_name == 'GEO':
        dir = custom_dir if custom_dir else './data/geo'

        if gen:
          build_image(dir)

        self.data_train, self.data_val, self.data_test =\
            load_geo(dir, val_percent=val_percent, test_percent=test_percent)

      else:
        raise RuntimeError(
            f'data_name {self.data_name} not recognized')
    except:
      print('Error loading dataset')
      pass

  def get_tf_data(self):
    try:
      if self.data_name == 'GEO':
        data_train, data_val, data_test = generate_tf_geo(
            self.data_train, self.data_val, self.data_test,
            self.conf['batch_size']
        )
      else:
        raise RuntimeError(f'data_name {self.data_name} not recognized')
    except:
      print('Error loading dataset')
      pass

    return data_train, data_val, data_test
