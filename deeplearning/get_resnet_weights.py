import sys,os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Lambda, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.initializers import glorot_uniform
from resnet import Resnet

def main(argv):
  infile = argv[0]
  outfile = argv[1]

  seqlen = 1500
  channels = 6
  maxnorm = 20.0

  # Create model
  resnet_model = Resnet(input_shape=(seqlen, channels), norm_max=maxnorm)
  samp1 = Input(shape=(seqlen, channels))
  enc_samp1 = resnet_model(samp1)
  samp2 = Input(shape=(seqlen, channels))
  enc_samp2 = resnet_model(samp2)
  diff_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
  diff_enc = diff_layer([enc_samp1, enc_samp2])

  dense_out = Dense(50,activation='relu',
                 kernel_constraint=MaxNorm(maxnorm,axis=[0,1]),
                 bias_constraint=MaxNorm(maxnorm,axis=0),
                 kernel_initializer=glorot_uniform(seed=0))(diff_enc)
  dense_out = Dropout(rate=0.2)(dense_out)
  output = Dense(1,activation='sigmoid',
                 kernel_constraint=MaxNorm(maxnorm,axis=[0,1]),
                 bias_constraint=MaxNorm(maxnorm,axis=0),
                 kernel_initializer=glorot_uniform(seed=0))(dense_out)
  model = Model(inputs=[samp1,samp2], outputs=output)
  model.load_weights(infile)
  for layer in model.layers:
    if layer.name == "model":
      resnet_model.set_weights(layer.get_weights())
      resnet_model.save_weights(outfile)

if __name__ == "__main__":
  main(sys.argv[1:])
