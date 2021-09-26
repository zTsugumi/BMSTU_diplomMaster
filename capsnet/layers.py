import tensorflow as tf

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
    norm_squared = tf.square(norm)
    return norm_squared / (1.0 + norm_squared) / norm * s


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
        self.Conv = tf.keras.layers.Conv2D(
            filters=self.C*self.L,
            kernel_size=self.k,
            strides=self.s,
            kernel_initializer='glorot_uniform',
            padding='valid',
            name='conv'
        )
        self.bias = self.add_weight(
            shape=(self.C, self.L),
            initializer='zeros',
            name='bias'
        )
        self.built = True

    def call(self, input):
        x = self.Conv(input)
        H, W = x.shape[1:3]
        x = tf.keras.layers.Reshape((H, W, self.C, self.L))(x)
        x /= self.C
        x += self.bias
        x = squash(x)
        return x


class DigitCaps(tf.keras.layers.Layer):
    '''
    This constructs a digit capsule layer
    '''

    def __init__(self, C, L, r, **kwargs):
        super(DigitCaps, self).__init__(**kwargs)
        self.C = C      # C: number of digit capsules
        self.L = L      # L: digit capsules dimension (num of properties)
        self.r = r      # r: number of routing

    def get_config(self):
        config = {
            'C': self.C,
            'L': self.L,
            'r': self.r
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
            shape=(H*W*input_C, input_L, self.C*self.L),
            initializer='glorot_uniform',
            name='W'
        )
        self.bias = self.add_weight(
            shape=(self.C, self.L),
            initializer='zeros',
            name='bias'
        )
        self.built = True

    def call(self, input):
        H, W, input_C, input_L = input.shape[1:]
        u = tf.reshape(input, shape=(
            -1, H*W*input_C, input_L))              # (None, H*W*input_C, input_L)

        # Here we multiply (1,8) x (8,160)
        u_hat = tf.einsum(
            '...ij,ijk->...ik', u, self.W)          # (None, H*W*input_C, C*L)
        u_hat = tf.reshape(u_hat, shape=(
            -1, H*W*input_C, self.C, self.L))       # (None, H*W*input_C, C, L)

        # Routing
        b = tf.zeros(
            tf.shape(u_hat)[:-1])[..., None]        # (None, H*W*input_C, C, 1)
        for r in range(self.r):
            c = tf.nn.softmax(b, axis=2)            # (None, H*W*input_C, C, 1)
            s = tf.reduce_sum(
                u_hat*c, axis=1, keepdims=True)     # (None, 1, C, L)
            s += self.bias
            v = squash(s)                           # (None, 1, C, L)
            if r < self.r - 1:
                agreement = tf.reduce_sum(
                    u_hat * v, axis=-1, keepdims=True)
                b += agreement
        v = tf.squeeze(v, axis=1)                   # (None, C, L)
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
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:
            return tuple([None, input_shape[1] * input_shape[2]])

    def call(self, input):
        if type(input) is list:
            input, mask = input
        else:
            x = safe_norm(input, axis=-1, keepdims=False)
            mask = tf.one_hot(
                tf.argmax(x, axis=1), depth=x.get_shape().as_list()[1]
            )
        masked = tf.keras.backend.batch_flatten(
            input * tf.expand_dims(mask, axis=-1))
        return masked
