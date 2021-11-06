import tensorflow as tf

CIFAR10_CLASS = 10


def preprocess_cifar10(image, label):
  image = tf.cast(image, dtype=tf.float32) / 255.0
  label = tf.one_hot(tf.squeeze(label), depth=CIFAR10_CLASS)
  image = tf.image.per_image_standardization(image)

  return image, label


def generator(image, label):
  return (image, label), (label, image)
