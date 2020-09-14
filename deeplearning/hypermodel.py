from kerastuner import HyperModel

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam

from metrics import macro_f1
from losses import focal_loss
from resnet import Resnet

class ResnetHyperModel(HyperModel):

  def __init__(self, hyperparam, seqlen, channels, pretrained_wts=None, num_classes=2):
    self.hyperparam = hyperparam
    self.seqlen = seqlen
    self.channels = channels
    self.num_classes = num_classes
    self.pretrained_weights = pretrained_wts

  def build(self, hp):
    maxnorm = hp.Choice('maxnorm', values=self.hyperparam['maxnorm'])

    resnet_model = Resnet(input_shape=(self.seqlen, self.channels), norm_max=maxnorm)
    if self.pretrained_weights is not None:
      resnet_model.set_weights(self.pretrained_weights)
    inp = Input(shape=(self.seqlen, self.channels))
    enc_inp = resnet_model(inp)

    dense_units = hp.Int('preclassification', min_value = self.hyperparam['dense_units']['min'],\
                         max_value = self.hyperparam['dense_units']['max'], step = self.hyperparam['dense_units']['step'])
    dense_out = Dense(units = dense_units, activation='relu',
                 kernel_constraint=MaxNorm(maxnorm,axis=[0,1]),
                 bias_constraint=MaxNorm(maxnorm,axis=0),
                 kernel_initializer=glorot_uniform(seed=0))(enc_inp)
    dense_out = Dropout(rate=hp.Choice('dropout', values = self.hyperparam['dropout']))(dense_out)
    output = Dense(self.num_classes, activation='softmax',
                 kernel_constraint=MaxNorm(maxnorm,axis=[0,1]),
                 bias_constraint=MaxNorm(maxnorm,axis=0),
                 kernel_initializer=glorot_uniform(seed=0))(dense_out)
    model = Model(inputs=inp, outputs=output)

    model.compile(optimizer=Adam(lr=hp.Choice('lr', values = self.hyperparam['lr'])),
                  loss=focal_loss(), metrics=['accuracy', macro_f1])

    return model

