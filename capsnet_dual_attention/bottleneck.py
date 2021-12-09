import tensorflow as tf
import tensorflow.keras as keras
from utils import Conv2DDown, Conv2DUp


class BottleNeck(keras.layers.Layer):
  '''Squeeze and Expand
  '''

  def __init__(self, n_block, block_base, max_features, **kwargs):
    super(BottleNeck, self).__init__(**kwargs)

    self.down_blocks = []
    for i in range(n_block):
      self.down_blocks.append(
          Conv2DDown(
              min(max_features, block_base * (2 ** (i + 1))),
              kernel_size=3,
              strides=2,
          )
      )

    self.up_blocks = []
    for i in range(n_block)[::-1]:
      self.up_blocks.append(
          Conv2DUp(
              min(max_features, block_base * (2 ** i)),
              kernel_size=3,
              strides=2,
          )
      )

  def get_config(self):
    config = {
        'down_blocks': self.down_blocks,
        'up_blocks': self.up_blocks
    }
    base_config = super(BottleNeck, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, input):
    # Encoder
    out_enc = [input]
    for block in self.down_blocks:
      out_enc.append(block(out_enc[-1]))

    # Decoder
    out_dec = out_enc.pop()
    for block in self.up_blocks:
      out_dec = block(out_dec)
      skip = out_enc.pop()
      out_dec = tf.concat([out_dec, skip], axis=-1)

    return out_dec
