import tensorflow as tf


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
