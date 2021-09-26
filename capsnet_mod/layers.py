import tensorflow as tf
import numpy as np

EPS = tf.keras.backend.epsilon()


def safe_norm(s, axis=-1, keepdims=True):
    '''
    Calculation of norm as tf.norm(), but here we add a small value of eps 
    to the result to avoid 0
    '''
    s_ = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keepdims)
    return tf.sqrt(s_ + EPS)


def squash(s):
    '''
    Squash activation
    '''
    norm = safe_norm(s, axis=-1)
    return (1.0 - 1.0/tf.exp(norm)) * (s / norm)


class PrimaryCaps(tf.keras.layers.Layer):
    '''
    This constructs a primary capsule layer using regular convolution layer
    '''

    def __init__(self, C, L, k, s, **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)
        self.C = C      # C: number of primary capsules
        self.L = L      # L: primary capsules dimension (num of properties)
        self.k = k      # k: kernel dimension
        self.s = s      # s: stride

    def get_config(self):
        config = {
            'C': self.C,
            'L': self.L,
            'k': self.k,
            's': self.s,
        }
        base_config = super(PrimaryCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        H, W = input_shape.shape[1:3]
        return (None, (H - self.k)/self.s + 1, (W - self.k)/self.s + 1, self.C, self.L)

    def build(self, input_shape):
        self.DW_Conv = tf.keras.layers.Conv2D(
            filters=self.C*self.L,
            kernel_size=self.k,
            strides=self.s,
            kernel_initializer='glorot_uniform',
            padding='valid',
            groups=self.C*self.L,
            activation='relu',
            name='conv'
        )
        self.built = True

    def call(self, input):
        x = self.DW_Conv(input)
        H, W = x.shape[1:3]
        x = tf.keras.layers.Reshape((H, W, self.C, self.L))(x)
        x = squash(x)
        return x


class DigitCaps(tf.keras.layers.Layer):
    '''
    This contructs the modified digit capsule layer
    '''

    def __init__(self, C, L, **kwargs):
        super(DigitCaps, self).__init__(**kwargs)
        self.C = C      # C: number of digit capsules
        self.L = L      # L: digit capsules dimension (num of properties)

    def get_config(self):
        config = {
            'C': self.C,
            'L': self.L
        }
        base_config = super(DigitCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (None, self.C, self.L)

    def build(self, input_shape):
        H = input_shape[1]
        W = input_shape[2]
        input_C = input_shape[3]
        input_L = input_shape[4]

        self.W = self.add_weight(               # Transformation matrix
            shape=(self.C, H*W*input_C, input_L, self.L),
            initializer='glorot_uniform',
            name='W'
        )
        self.bias = self.add_weight(               # Coupling Coefficient
            shape=(self.C, H*W*input_C, 1),
            initializer='zeros',
            name='bias'
        )

    def call(self, input):
        H, W, input_C, input_L = input.shape[1:]
        u = tf.reshape(input, shape=(
            -1, H*W*input_C, input_L))

        u_hat = tf.einsum(
            '...ij,kijl->...kil', u, self.W)

        a = tf.einsum(                          # Calculate attention
            '...ij,...kj->...i', u_hat, u_hat)[..., None]
        a = a / tf.sqrt(tf.cast(self.L, tf.float32))
        a = tf.nn.softmax(a, axis=1)

        # print(u_hat.shape)
        # print(a.shape)
        # tf.print(tf.squeeze(safe_norm(u_hat)), summarize=-1, output_stream='file://test.txt')
        # tf.print(tf.squeeze(a), summarize=-1, output_stream='file://test1.txt')

        s = tf.reduce_sum(u_hat*(a + self.bias), axis=-2)
        v = squash(s)

        return v


class Length(tf.keras.layers.Layer):
    '''
    This constructs the computation of the length of each capsule in a layer
    '''

    def get_config(self):
        base_config = super(Length, self).get_config()
        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def call(self, input):
        return safe_norm(input, axis=-1, keepdims=False)


class Mask(tf.keras.layers.Layer):
    '''
    This constructs the mask operation
    '''

    def get_config(self):
        base_config = super(Mask, self).get_config()
        return base_config

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            return tuple([None, input_shape[0][2]])
        else:
            return tuple([None, input_shape[2]])

    def call(self, input):
        if type(input) is list:
            input, mask = input
            mask = tf.argmax(mask, axis=1)
        else:
            x = safe_norm(input, axis=-1, keepdims=False)
            mask = tf.argmax(x, axis=1)

        idx = tf.range(start=0, limit=tf.shape(input)[0], delta=1)
        idx = tf.stack([idx, tf.cast(mask, tf.int32)], axis=1)

        masked = tf.gather_nd(input, idx)

        return masked
