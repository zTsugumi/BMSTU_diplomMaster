import tensorflow as tf

N_THREAD = 8
CIFAR10_SIZE = 50000


def generator(image, label):
  return (image, label), (label, image)


def generate_tf_cifar10(x_train, y_train, x_val, y_val, x_test, y_test, batch_size):
  data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  data_train = data_train.shuffle(buffer_size=CIFAR10_SIZE)
  data_train = data_train.map(generator, num_parallel_calls=N_THREAD)
  data_train = data_train.batch(batch_size, drop_remainder=True)
  data_train = data_train.prefetch(-1)

  data_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
  data_val = data_train.map(generator, num_parallel_calls=N_THREAD)
  data_val = data_train.batch(batch_size, drop_remainder=True)
  data_val = data_train.prefetch(-1)

  data_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  data_test = data_test.cache()
  data_test = data_test.map(generator, num_parallel_calls=N_THREAD)
  data_test = data_test.batch(batch_size, drop_remainder=True)
  data_test = data_test.prefetch(-1)

  return data_train, data_val, data_test
