from tensorflow import keras
from tensorflow.keras import Input, Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Conv1D, LeakyReLU, Flatten,\
                                    Add, BatchNormalization, MaxPooling1D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.keras.constraints import UnitNorm, MaxNorm

def identity_block(inputs, filters, ksz, stage, block, norm_max=1.0):
  """
  Identity block for ResNet

  Parameters
  __________
  inputs : input tensor of shape (batch_size, num_steps_prev, num_ch_prev)
  filters : list of 3 integers defining num of conv filters in main path
  ksz : filter width of middle convolutional layer
  stage : integer used to name layers
  block : string used to name layers
  norm_max    : maximum norm for constraint

  Returns
  _______
  x - output of the identity block, tensor of shape (batch_size, num_steps, num_ch)
  """

  conv_base_name = 'res' + str(stage) + block + '_branch'
  bn_base_name = 'bn' + str(stage) + block + '_branch'

  F1, F2, F3 = filters

  # Shortcut path
  x_shortcut = inputs

  # Main path
  x = Conv1D(filters=F1, kernel_size=1, strides=1, padding='valid',
             name=conv_base_name + '2a', use_bias=False,
             kernel_constraint=MaxNorm(norm_max,axis=[0,1,2]),
             kernel_initializer=glorot_uniform(seed=0))(inputs)
  x = LeakyReLU(alpha=0.1)(x)
  x = BatchNormalization(axis=-1, momentum=0.9, name=bn_base_name + '2a',
                         gamma_constraint=MaxNorm(norm_max,axis=0),
                         beta_constraint=MaxNorm(norm_max,axis=0))(x)

  x = Conv1D(filters=F2, kernel_size=ksz, strides=1, padding='same',
             name=conv_base_name + '2b', use_bias=False,
             kernel_constraint=MaxNorm(norm_max,axis=[0,1,2]),
             kernel_initializer=glorot_uniform(seed=0))(x)
  x = LeakyReLU(alpha=0.1)(x)
  x = BatchNormalization(axis=-1, momentum=0.9, name=bn_base_name + '2b',
                         gamma_constraint=MaxNorm(norm_max,axis=0),
                         beta_constraint=MaxNorm(norm_max,axis=0))(x)

  x = Conv1D(filters=F3, kernel_size=1, strides=1, padding='valid',
             name=conv_base_name + '2c', use_bias=False,
             kernel_constraint=MaxNorm(norm_max,axis=[0,1,2]),
             kernel_initializer=glorot_uniform(seed=0))(x)

  x = Add()([x, x_shortcut])
  x = LeakyReLU(alpha=0.1)(x)
  x = BatchNormalization(axis=-1, momentum=0.9, name=bn_base_name + '2c',
                         gamma_constraint=MaxNorm(norm_max,axis=0),
                         beta_constraint=MaxNorm(norm_max,axis=0))(x)
  return x

def conv_block(inputs, filters, ksz, stage, block, s=2, norm_max=1.0):
  """
  Convolutional block for ResNet

  Parameters
  __________
  inputs : input tensor of shape (batch_size, num_steps_prev, num_ch_prev)
  filters : list of 3 integers defining num of conv filters in main path
  ksz : filter width of middle convolutional layer
  stage : integer used to name layers
  block : string used to name layers
  s : integer specifying stride
  norm_max    : maximum norm for constraint

  Returns
  _______
  x - output of the convolutional block, tensor of shape (batch_size, num_steps, num_ch)
  """

  conv_base_name = 'res' + str(stage) + block + '_branch'
  bn_base_name = 'bn' + str(stage) + block + '_branch'

  F1, F2, F3 = filters

  # Shortcut path
  x_shortcut = inputs
  x_shortcut = Conv1D(filters=F3, kernel_size=1, strides=s, padding='valid',
                      name=conv_base_name + '1', use_bias=False,
                      kernel_constraint=MaxNorm(norm_max,axis=[0,1,2]),
                      kernel_initializer=glorot_uniform(seed=0))(x_shortcut)

  # Main path
  x = Conv1D(filters=F1, kernel_size=1, strides=s, padding='valid',
             name=conv_base_name + '2a', use_bias=False,
             kernel_constraint=MaxNorm(norm_max,axis=[0,1,2]),
             kernel_initializer=glorot_uniform(seed=0))(inputs)
  x = LeakyReLU(alpha=0.1)(x)
  x = BatchNormalization(axis=-1, momentum=0.9, name=bn_base_name + '2a',
                         gamma_constraint=MaxNorm(norm_max,axis=0),
                         beta_constraint=MaxNorm(norm_max,axis=0))(x)

  x = Conv1D(filters=F2, kernel_size=ksz, strides=1, padding='same',
             name=conv_base_name + '2b', use_bias=False,
             kernel_constraint=MaxNorm(norm_max,axis=[0,1,2]),
             kernel_initializer=glorot_uniform(seed=0))(x)
  x = LeakyReLU(alpha=0.1)(x)
  x = BatchNormalization(axis=-1, momentum=0.9, name=bn_base_name + '2b',
                         gamma_constraint=MaxNorm(norm_max,axis=0),
                         beta_constraint=MaxNorm(norm_max,axis=0))(x)

  x = Conv1D(filters=F3, kernel_size=1, strides=1, padding='valid',
             name=conv_base_name + '2c', use_bias=False,
             kernel_constraint=MaxNorm(norm_max,axis=[0,1,2]),
             kernel_initializer=glorot_uniform(seed=0))(x)

  x = Add()([x, x_shortcut])
  x = LeakyReLU(alpha=0.1)(x)
  x = BatchNormalization(axis=-1, momentum=0.9, name=bn_base_name + '2c',
                         gamma_constraint=MaxNorm(norm_max,axis=0),
                         beta_constraint=MaxNorm(norm_max,axis=0))(x)
  return x

def FCN(input_shape, norm_max=1.0):
  """
  Generate a fully convolutional neural network (FCN) model.

  Parameters
  ----------
  input_shape : tuple defining shape of the input dataset: (num_timesteps, num_channels)
  norm_max    : maximum norm for constraint

  Returns
  -------
  model : Keras model
  """
  inputs = Input(shape = input_shape)

  # Stage 1
  x = Conv1D(filters=32, kernel_size=7, strides=2,
             padding='valid', use_bias=False,
             kernel_constraint=MaxNorm(norm_max, axis=[0,1,2]),
             name = 'conv1', kernel_initializer=glorot_uniform(seed=0))(inputs)
  x = LeakyReLU(alpha=0.1)(x)
  x = BatchNormalization(axis=-1, momentum=0.9, name='bn_conv1',
                         gamma_constraint=MaxNorm(norm_max,axis=0),
                         beta_constraint=MaxNorm(norm_max,axis=0))(x)

  # Stage 2
  x = conv_block(x, ksz=3, filters=[16,16,32], stage=2, block='a', s=2, norm_max=norm_max)
  x = identity_block(x, ksz=3, filters=[16,16,32], stage=2, block='b', norm_max=norm_max)
  x = identity_block(x, ksz=3, filters=[16,16,32], stage=2, block='c', norm_max=norm_max)

#  # Stage 3
#  x = conv_block(x, ksz=3, filters=[32,32,64], stage=3, block='a', s=2, norm_max=norm_max)
#  x = identity_block(x, ksz=3, filters=[32,32,64], stage=3, block='b', norm_max=norm_max)
#  x = identity_block(x, ksz=3, filters=[32,32,64], stage=3, block='c', norm_max=norm_max)
#  x = identity_block(x, ksz=3, filters=[32,32,64], stage=3, block='d', norm_max=norm_max)
#
#  # Stage 4
#  x = conv_block(x, ksz=3, filters=[128,128,256], stage=4, block='a', s=2)
#  x = identity_block(x, ksz=3, filters=[128,128,256], stage=4, block='b')
#  x = identity_block(x, ksz=3, filters=[128,128,256], stage=4, block='c')
#  x = identity_block(x, ksz=3, filters=[128,128,256], stage=4, block='d')
#  x = identity_block(x, ksz=3, filters=[128,128,256], stage=4, block='e')

#  # Stage 5
#  x = conv_block(x, ksz=3, filters=[256,256,512], stage=5, block='a', s=2)
#  x = identity_block(x, ksz=3, filters=[256,256,512], stage=5, block='b')
#  x = identity_block(x, ksz=3, filters=[256,256,512], stage=5, block='c')
#  x = identity_block(x, ksz=3, filters=[256,256,512], stage=5, block='d')
#  x = identity_block(x, ksz=3, filters=[256,256,512], stage=5, block='e')
#  x = identity_block(x, ksz=3, filters=[256,256,512], stage=5, block='f')

  # Output stage
  x = MaxPooling1D(pool_size=2)(x)
  outputs = Flatten(name='last_layer')(x)

  model = Model(inputs=inputs, outputs=outputs)
  return model
