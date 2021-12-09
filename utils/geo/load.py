import os
import numpy as np
from utils.ops_image import imread, imresize


def load_geo(
        dir='./data/geo',
        dim=(128, 128),
        val_percent=0.1,
        test_percent=0.1):
  data = open(os.path.join(dir, 'demo.txt')).read().splitlines()[:10000]
  data_size = len(data)

  datas = []
  for idx in range(data_size):
    images = []
    for k in range(2):
      image = imread(os.path.join(dir, f'{data[idx]}_im{k + 1}.png'))
      image = imresize(image, size=dim)
      images.append(image)
    datas.append(images)

  test_index = int(data_size * (1 - test_percent))
  val_index = int(test_index * (1 - val_percent))
  data_train = np.array(datas[:val_index])
  data_val = np.array(datas[val_index:test_index])
  data_test = np.array(datas[test_index:])

  data_train = data_train.transpose(0, 1, 3, 4, 2)
  data_val = data_val.transpose(0, 1, 3, 4, 2)
  data_test = data_test.transpose(0, 1, 3, 4, 2)

  return data_train, data_val, data_test
