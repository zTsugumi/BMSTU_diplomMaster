import numpy as np
import pickle
import tarfile
import os

TAR_NAME = 'cifar-10-python.tar.gz'
CIFAR10_SIZE = 50000
TEST_SIZE = 10000
BATCH_SIZE = 10000


def load_cifar10(
    dir='./data/cifar10',
    val_percent=0.1,
):
  n_train = int(CIFAR10_SIZE * (1 - val_percent))
  n_val = CIFAR10_SIZE - n_train
  val_index = int(BATCH_SIZE * (1 - val_percent))

  x_train = np.empty((n_train, 32, 32, 3), dtype='uint8')
  y_train = np.empty((n_train,), dtype='uint8')
  x_val = np.empty((n_val, 32, 32, 3), dtype='uint8')
  y_val = np.empty((n_val,), dtype='uint8')

  fpath = os.path.join(dir, TAR_NAME)
  with tarfile.open(fpath)as f:
    f.extractall(dir)
    f.close()

  for i in range(1, 6):
    fpath = os.path.join(dir, 'cifar-10-batches-py/data_batch_' + str(i))
    with open(fpath, mode='rb') as f:
      batch = pickle.load(f, encoding='latin1')

    data = batch['data']\
        .reshape((len(batch['data']), 3, 32, 32))\
        .transpose(0, 2, 3, 1)
    labels = batch['labels']

    # split train, val
    x_train[(i - 1) * val_index: i * val_index, :, :, :] = data[:val_index]
    y_train[(i - 1) * val_index: i * val_index] = labels[:val_index]

    x_val[(i - 1) * (BATCH_SIZE - val_index): i * (BATCH_SIZE - val_index), :, :, :] = data[val_index:]
    y_val[(i - 1) * (BATCH_SIZE - val_index): i * (BATCH_SIZE - val_index)] = labels[val_index]

  fpath = os.path.join(dir, 'cifar-10-batches-py/test_batch')
  with open(fpath, mode='rb') as f:
    batch = pickle.load(f, encoding='latin1')

  x_test = batch['data']\
      .reshape((len(batch['data']), 3, 32, 32))\
      .transpose(0, 2, 3, 1)
  y_test = batch['labels']

  y_train = np.reshape(y_train, (len(y_train), 1))
  y_val = np.reshape(y_val, (len(y_val), 1))
  y_test = np.reshape(y_test, (len(y_test), 1))

  return (x_train, y_train), (x_val, y_val), (x_test, y_test)
