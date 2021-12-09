import imageio
import numpy as np
import collections
import cv2
import shutil
import os


def mkdir(path, clean=False):
  if clean and os.path.exists(path):
    shutil.rmtree(path)
  if not os.path.exists(path):
    os.makedirs(path)


def imretype(im, dtype):
  im = np.array(im)

  if im.dtype in ['float', 'float16', 'float32', 'float64']:
    im = im.astype(np.float)
  elif im.dtype == 'uint8':
    im = im.astype(np.float) / 255.
  elif im.dtype == 'uint16':
    im = im.astype(np.float) / 65535.
  else:
    raise NotImplementedError(
        'unsupported source dtype: {0}'.format(im.dtype))

  assert np.min(im) >= 0 and np.max(im) <= 1

  if dtype in ['float', 'float16', 'float32', 'float64']:
    im = im.astype(dtype)
  elif dtype == 'uint8':
    im = (im * 255.).astype(dtype)
  elif dtype == 'uint16':
    im = (im * 65535.).astype(dtype)
  else:
    raise NotImplementedError(
        'unsupported target dtype: {0}'.format(dtype))

  return im


def imresize(im, size):
  cfirst = im.shape[0] in [1, 3, 4]
  if np.ndim(im) == 3 and cfirst:
    im = im.transpose(1, 2, 0)

  im = cv2.resize(im, dsize=size)

  if np.ndim(im) == 3 and cfirst:
    im = im.transpose(2, 0, 1)
  return im


def imread(path, size=None, dtype='float32', cfirst=True):
  im = imageio.imread(path)
  im = imretype(im, dtype=dtype)

  if size is not None:
    im = imresize(im, size=size)

  if np.ndim(im) == 3 and cfirst:
    im = im.transpose(2, 0, 1)
  return im


def imwrite(path, obj):
  if not isinstance(obj, (collections.Sequence, collections.UserList)):
    obj = [obj]
  writer = imageio.get_writer(path)
  for im in obj:
    im = imretype(im, dtype='uint8').squeeze()
    if len(im.shape) == 3 and im.shape[0] == 3:
      im = np.transpose(im, (1, 2, 0))
    writer.append_data(im)
  writer.close()
