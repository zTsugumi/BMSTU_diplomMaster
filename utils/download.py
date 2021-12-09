import traceback
import time
import os
import requests
from functools import wraps
from tqdm import tqdm


def retry(ExceptionToCheck, tries=4, delay=3, backoff=2):
  def deco_retry(f):
    @wraps(f)
    def f_retry(*args, **kwargs):
      mtries, mdelay = tries, delay
      while mtries > 1:
        try:
          return f(*args, **kwargs)
        except KeyboardInterrupt as e:
          raise e
        except ExceptionToCheck as e:
          print('%s, retrying in %d seconds...' % (str(e), mdelay))
          traceback.print_exc()
          time.sleep(mdelay)
          mtries -= 1
          mdelay *= backoff
      return f(*args, **kwargs)

    return f_retry  # true decorator

  return deco_retry


@retry(Exception)
def download_file(url, path):
  r = requests.get(url, stream=True)
  total_size = int(r.headers.get('content-length'))

  if os.path.exists(path) and os.path.getsize(path) == total_size:
    return

  chunk_size = 4 * 1024 * 1024
  buff_size = 16 * 1024 * 1024
  bar = tqdm(
      total=total_size,
      unit='B',
      unit_scale=True,
      unit_divisor=1024,
      desc=os.path.split(path)[-1])
  imcomplete_download = False

  try:
    with open(path, 'wb', buffering=buff_size) as f:
      for chunk in r.iter_content(chunk_size=chunk_size):
        datasize = f.write(chunk)
        bar.update(datasize)
  except Exception as e:
    raise e
  finally:
    bar.close()
    if os.path.exists(path) and os.path.getsize(path) != total_size:
      imcomplete_download = True
      os.remove(path)

  if imcomplete_download:
    raise Exception('Incomplete download')


def sequential_downloader(url, fns, data_dir):
  os.makedirs(data_dir, exist_ok=True)

  for fn in fns:
    fpath = os.path.join(data_dir, fn)
    furl = url + fn
    download_file(furl, fpath)


def download_lfw(data_dir='./data/lfw'):
  sequential_downloader(
      'http://vis-www.cs.umass.edu/lfw/',
      [
          'lfw-deepfunneled.tgz',
          'lfw.tgz',
      ],
      data_dir)


def download_cifar10(data_dir='./data/cifar10'):
  sequential_downloader(
      'https://www.cs.toronto.edu/~kriz/',
      [
          'cifar-10-python.tar.gz'
      ],
      data_dir)


# def download_voxceleb(data_dir='./data/voxceleb'):
#   sequential_downloader(
#       'https://www.cs.toronto.edu/~kriz/',
#       [
#           'cifar-10-python.tar.gz'
#       ],
#       data_dir)


# def download_exercise(data_dir='./data/exercise'):
#   sequential_downloader(
#       'http://psd.csail.mit.edu/models/',
#       [
#           'snapshot.pth'
#       ],
#       data_dir)
