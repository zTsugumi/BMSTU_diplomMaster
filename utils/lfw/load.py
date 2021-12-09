import numpy as np
import os
import cv2
import pandas as pd
import tarfile


ATTR_NAME = '/lfw_attributes.txt'
IMGS_NAME = '/lfw-deepfunneled.tgz'
RAW_IMGS_NAME = '/lfw.tgz'


def decode_image_from_raw_bytes(raw_bytes):
  img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img


def load_lfw(
        dir='./data/lfw',
        use_raw=False,
        dx=80, dy=80,
        dimx=45, dimy=45):
  attr_dir = dir + ATTR_NAME
  imgs_dir = dir + IMGS_NAME
  raw_imgs_dir = dir + RAW_IMGS_NAME

  # read attrs
  df_attrs = pd.read_csv(attr_dir, sep='\t', skiprows=1)
  df_attrs.columns = list(df_attrs.columns)[1:] + ['NaN']
  df_attrs = df_attrs.drop('NaN', axis=1)
  imgs_with_attrs = set(map(tuple, df_attrs[['person', 'imagenum']].values))

  # read photos
  all_photos = []
  photo_ids = []

  with tarfile.open(raw_imgs_dir if use_raw else imgs_dir) as f:
    for m in f.getmembers():
      if m.isfile() and m.name.endswith('.jpg'):
        # prepare image
        img = decode_image_from_raw_bytes(f.extractfile(m).read())
        img = img[dy:-dy, dx:-dx]
        img = cv2.resize(img, (dimx, dimy))
        # parse person
        fname = os.path.split(m.name)[-1]
        fname_splitted = fname[:-4].replace('_', ' ').split()
        person_id = ' '.join(fname_splitted[:-1])
        photo_number = int(fname_splitted[-1])
        if (person_id, photo_number) in imgs_with_attrs:
          all_photos.append(img)
          photo_ids.append(
              {'person': person_id, 'imagenum': photo_number})

  photo_ids = pd.DataFrame(photo_ids)
  all_photos = np.stack(all_photos).astype('uint8')

  # preserve photo_ids order!
  all_attrs = photo_ids.merge(df_attrs, on=('person', 'imagenum')).drop([
      'person', 'imagenum'], axis=1)

  return all_photos, all_attrs
