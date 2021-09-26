import tensorflow as tf

CIFAR10_CLASS = 10
CIFAR10_SIZE = 50000
N_THREAD = 8


def pre_process(image, label):
    image = tf.cast(image, dtype=tf.float32) / 255.0
    label = tf.one_hot(tf.squeeze(label), depth=CIFAR10_CLASS)
    image = tf.image.per_image_standardization(image)

    return image, label


def generator(image, label):
    return (image, label), (label, image)


def generate_tf_data(x_train, y_train, x_test, y_test, batch_size):
    data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    data_train = data_train.shuffle(buffer_size=CIFAR10_SIZE)
    data_train = data_train.map(generator, num_parallel_calls=N_THREAD)
    data_train = data_train.batch(batch_size, drop_remainder=True)
    data_train = data_train.prefetch(-1)

    data_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    data_test = data_test.cache()
    data_test = data_test.map(generator, num_parallel_calls=N_THREAD)
    data_test = data_test.batch(batch_size, drop_remainder=True)
    data_test = data_test.prefetch(-1)

    return data_train, data_test
