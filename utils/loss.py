import tensorflow as tf


def L2_loss(img_in, img_out):
  diff = img_in - img_out
  L = tf.sqrt(tf.reduce_sum(diff ** 2))

  return L


def L1_loss(img_in, img_out):
  b = tf.shape(img_in)[0]
  img_in = tf.reshape(img_in, (b, -1))
  img_out = tf.reshape(img_out, (b, -1))

  L1 = tf.abs(img_in - img_out)
  L = tf.reduce_mean(tf.reduce_sum(L1, axis=-1))

  return L


def riemannian_loss(img_in, img_out):
  # gamma = 0.1
  dist = tf.math.log(img_in) - tf.math.log(img_out)
  L = tf.reduce_mean(tf.exp(tf.abs(dist)))

  return L


def margin_loss(y_true, y_pred):
  lamda = 0.5
  m_plus = 0.9
  m_minus = 0.1

  margin_left = tf.square(tf.maximum(0.0, m_plus - y_pred))
  margin_right = tf.square(tf.maximum(0.0, y_pred - m_minus))

  margin_left = y_true * margin_left
  margin_right = lamda * (1.0 - y_true) * margin_right

  L = margin_left + margin_right
  L = tf.reduce_mean(tf.reduce_sum(L, axis=-1))

  return L
