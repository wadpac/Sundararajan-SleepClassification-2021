from tensorflow import keras
from tensorflow.keras import Input, Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Conv1D, Lambda, Conv2DTranspose, LeakyReLU,\
                                    Add, BatchNormalization, GlobalAveragePooling1D, ZeroPadding1D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.keras.constraints import UnitNorm

def identity_block(inputs, filters, ksz, stage, block, activation='relu'):
  """
  Identity block for ResNet

  Parameters
  __________
  inputs : input tensor of shape (batch_size, num_steps_prev, num_ch_prev)
  filters : list of 3 integers defining num of conv filters in main path
  ksz : filter width of middle convolutional layer
  stage : integer used to name layers
  block : string used to name layers
  activation : activation function to be used

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
             name=conv_base_name + '2a',
             kernel_constraint=UnitNorm(axis=[0,1,2]), use_bias=False,
             kernel_initializer=glorot_uniform(seed=0))(inputs)
  x = LeakyReLU(alpha=0.1)(x)
  x = BatchNormalization(axis=-1, name=bn_base_name + '2a')(x)

  x = Conv1D(filters=F2, kernel_size=ksz, strides=1, padding='same',
             name=conv_base_name + '2b',
             kernel_constraint=UnitNorm(axis=[0,1,2]), use_bias=False,
             kernel_initializer=glorot_uniform(seed=0))(x)
  x = LeakyReLU(alpha=0.1)(x)
  x = BatchNormalization(axis=-1, name=bn_base_name + '2b')(x)

  x = Conv1D(filters=F3, kernel_size=1, strides=1, padding='valid',
             name=conv_base_name + '2c',
             kernel_constraint=UnitNorm(axis=[0,1,2]), use_bias=False,
             kernel_initializer=glorot_uniform(seed=0))(x)
  x = LeakyReLU(alpha=0.1)(x)
  x = BatchNormalization(axis=-1, name=bn_base_name + '2c')(x)

  x = Add()([x, x_shortcut])
  x = LeakyReLU(alpha=0.1)(x)
  return x

def conv_block(inputs, filters, ksz, stage, block, s=2, activation='relu'):
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
  activation : activation function to be used

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
                      name=conv_base_name + '1',
                      kernel_constraint=UnitNorm(axis=[0,1,2]), use_bias=False,
                      kernel_initializer=glorot_uniform(seed=0))(x_shortcut)
  x_shortcut = BatchNormalization(axis=-1, name=bn_base_name + '1')(x_shortcut)

  # Main path
  x = Conv1D(filters=F1, kernel_size=1, strides=s, padding='valid',
             name=conv_base_name + '2a',
             kernel_constraint=UnitNorm(axis=[0,1,2]), use_bias=False,
             kernel_initializer=glorot_uniform(seed=0))(inputs)
  x = LeakyReLU(alpha=0.1)(x)
  x = BatchNormalization(axis=-1, name=bn_base_name + '2a')(x)

  x = Conv1D(filters=F2, kernel_size=ksz, strides=1, padding='same',
             name=conv_base_name + '2b',
             kernel_constraint=UnitNorm(axis=[0,1,2]), use_bias=False,
             kernel_initializer=glorot_uniform(seed=0))(x)
  x = LeakyReLU(alpha=0.1)(x)
  x = BatchNormalization(axis=-1, name=bn_base_name + '2b')(x)

  x = Conv1D(filters=F3, kernel_size=1, strides=1, padding='valid',
             name=conv_base_name + '2c',
             kernel_constraint=UnitNorm(axis=[0,1,2]), use_bias=False,
             kernel_initializer=glorot_uniform(seed=0))(x)
  x = LeakyReLU(alpha=0.1)(x)
  x = BatchNormalization(axis=-1, name=bn_base_name + '2c')(x)

  x = Add()([x, x_shortcut])
  x = LeakyReLU(alpha=0.1)(x)
  return x

def Conv1DTranspose(inputs, filters, ksz, s=2, padding='same'):
  """
  1D Transposed convolution for FCN

  Parameters
  __________
  inputs : input tensor of shape (batch_size, num_steps_prev, num_ch_prev)
  filters : integer defining num of conv filters
  ksz : filter width of convolutional layer
  s : integer specifying stride
  padding : padding for the convolutional layer

  Returns
  _______
  x : output tensor of shape (batch_size, num_steps, num_ch)
  """
  x = Lambda(lambda x: K.expand_dims(x, axis=2))(inputs)
  x = Conv2DTranspose(filters=filters, kernel_size=(ksz, 1), strides=(s, 1), padding=padding,
                      name='conv_transpose', use_bias=False,
                      kernel_constraint=UnitNorm(axis=[0,1,2]),
                      kernel_initializer=glorot_uniform(seed=0))(x)
  x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
  return x

def FCN(input_shape, max_seqlen, num_classes=2, activation='relu'):
  """
  Generate a fully convolutional neural network (FCN) model.

  Parameters
  ----------
  input_shape : tuple defining shape of the input dataset: (num_timesteps, num_channels)
  num_classes : integer defining number of classes for classification task
  activation : activation function to be used

  Returns
  -------
  model : Keras model
  """
  outputdim = num_classes  # number of classes

  inputs = Input(shape = input_shape)

  # Zero padding
  pad_wd = (max_seqlen - input_shape[0])//2
  x = ZeroPadding1D(pad_wd)(inputs)

  # Stage 1
  x = Conv1D(filters=32, kernel_size=7, strides=2, padding='valid',
             kernel_constraint=UnitNorm(axis=[0,1,2]), use_bias=False,
             name = 'conv1', kernel_initializer=glorot_uniform(seed=0))(x)
  x = LeakyReLU(alpha=0.1)(x)
  x = BatchNormalization(axis=-1, name='bn_conv1')(x)

  # Stage 2
  x = conv_block(x, ksz=3, filters=[16,16,32], stage=2, block='a', s=2)
  x = identity_block(x, ksz=3, filters=[16,16,32], stage=2, block='b')
  x = identity_block(x, ksz=3, filters=[16,16,32], stage=2, block='c')

#  # Stage 3
#  x = conv_block(x, ksz=3, filters=[64,64,128], stage=3, block='a', s=2)
#  x = identity_block(x, ksz=3, filters=[64,64,128], stage=3, block='b')
#  x = identity_block(x, ksz=3, filters=[64,64,128], stage=3, block='c')
#  x = identity_block(x, ksz=3, filters=[64,64,128], stage=3, block='d')

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
  x = Conv1DTranspose(x, filters=64, ksz=5, s=4)
  x = GlobalAveragePooling1D()(x)
  outputs = Dense(num_classes, activation='softmax',
                  kernel_constraint=UnitNorm(axis=[0,1]), bias_constraint=UnitNorm(axis=0),
                  kernel_initializer=glorot_uniform(seed=0))(x)

  model = Model(inputs=inputs, outputs=outputs)
  return model
