import tensorflow as tf
from model import Model
from capsnet_dual_attention.gen_model import build_graph
from utils import checkpoint, lr_sched, L1_loss, L2_loss, riemannian_loss


class CapsNetDualAttention(Model):
  def __init__(self, name, mode, conf_path='conf.json'):
    Model.__init__(self, name, mode, conf_path)
    self.load_conf()

    self.dir_model = self.conf['dir_log'] + f'/capsnet_dual_attention_{name}'
    self.dir_log = self.conf['dir_log'] + f'/capsnet_dual_attention_{name}'

    self.model = build_graph(
        name, self.conf[f'input_{name}'], self.mode)

  def train(self, dataset, initial_epoch=0):
    data_train, data_val, _ = dataset.get_tf_data()

    cp, tb = checkpoint(self.dir_model, self.dir_log)
    lr = lr_sched(self.conf['lr'], self.conf['lr_decay'])

    self.model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=self.conf['lr']),
        loss=L2_loss,
    )

    if initial_epoch > 0:
      self.load_weight(initial_epoch)

    history = self.model.fit(
        data_train,
        validation_data=data_val,
        epochs=self.conf['epochs'],
        batch_size=self.conf['batch_size'],
        initial_epoch=initial_epoch,
        callbacks=[cp, tb, lr]
    )

    return history
