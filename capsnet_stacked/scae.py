import tensorflow as tf
from model import Model
from utils import checkpoint
from capsnet_stacked.gen_model import build_graph


class SCAE(Model):
  def __init__(self, name, mode, conf_path='conf.json'):
    Model.__init__(self, name, mode, conf_path)
    self.load_conf()

    self.dir_model = self.conf['dir_log'] + f'/scae_{name}'
    self.dir_log = self.conf['dir_log'] + f'/scae_{name}'

    self.model = build_graph(
        name, self.conf[f'input_{name}'], self.mode)
