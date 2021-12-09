import tensorflow as tf
import tensorflow.keras as keras
from capsnet_stacked.part_capsule import ImageDecoder, ImageEncoder

params_CIFAR10 = {
    {
        'params_part_encoder': {
            'n_caps': 16,
            'n_caps_dims': 6,
            'n_features': 16,
            'noise_scale': 4.
        },
        'params_part_decoder': {
            'img_size': [32]*2,
            'template_size': [11]*2,
            'n_channels': 1,
            'learn_output_scale': False,
            'colorize_templates': False,
            'use_alpha_channel': False,
        }
    }

}


def build_graph(name, input_shape, mode, verbose=False):
  '''
  This contructs the whole architecture of Capsule Network 
  (Encoder + Decoder)
  '''
  if name == 'CIFAR10':
    params = params_CIFAR10
  else:
    raise RuntimeError(f'name {name} not recognized')

  inputs = keras.Input(input_shape)

  part_encoder = ImageEncoder(params['params_part_encoder'], input_shape)
  pose, feature, presence, presence_logit = part_encoder(inputs)

  part_decoder = ImageDecoder(params['params_part_decoder'])
