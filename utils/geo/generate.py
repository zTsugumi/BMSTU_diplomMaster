import tensorflow as tf

GEO_SIZE = 1000


def generate_tf_geo(data_train, data_val, data_test, batch_size):
  data_train = tf.data.Dataset.from_tensor_slices((data_train[:, 0], data_train[:, 0]))
  data_train = data_train.shuffle(buffer_size=GEO_SIZE)
  data_train = data_train.batch(batch_size, drop_remainder=True)
  data_train = data_train.prefetch(-1)

  data_val = tf.data.Dataset.from_tensor_slices((data_val[:, 0], data_val[:, 0]))
  data_val = data_val.batch(batch_size, drop_remainder=True)
  data_val = data_val.prefetch(-1)

  data_test = tf.data.Dataset.from_tensor_slices(data_test[:, 0])
  data_test = data_test.cache()
  data_test = data_test.batch(batch_size, drop_remainder=True)

  return data_train, data_val, data_test
