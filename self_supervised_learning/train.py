import sys,os
import numpy as np
import random
import argparse
from collections import Counter

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.initializers import glorot_uniform

import matplotlib.pyplot as plt

from FCN import FCN
from datagenerator import DataGenerator
from callbacks import BatchRenormScheduler

np.random.seed(2)

# Limit GPU memory allocated
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  tf.config.experimental.set_memory_growth(gpus[0], True)

def plot_results(train_result, val_result, out_fname, metric='Loss'):
  plt.Figure()
  plt.plot(train_result, label='train')
  plt.plot(val_result, label='val')
  plt.title(metric)
  plt.ylabel(metric)
  plt.xlabel('Epochs')
  ylim = 1.0 if metric != 'Loss' else 5.0
  plt.ylim(0,ylim)
  plt.legend(['Train', 'Val'], loc='upper right')
  plt.savefig(out_fname)
  plt.clf()

def get_best_model(indir, mode='max'):
  files = os.listdir(indir)
  files = [fname for fname in files if fname.endswith('.h5')]
  metric = np.array([float(fname.split('.h5')[0].split('-')[1]) for fname in files])
  epoch = np.array([int(fname.split('.h5')[0].split('-')[0]) for fname in files])
  best_idx = np.argmax(metric) if mode == 'max' else np.argmin(metric)
  return files[best_idx], epoch[best_idx], metric[best_idx]

def main(argv):
  traindir = args.traindir
  valdir = args.valdir
  testdir = args.testdir
  outdir = args.outdir

  if not os.path.exists(outdir):
    os.makedirs(outdir)

  dirstr = 'lr{:.4f}-maxnorm{:.2f}-batchsize{:d}'.format(args.lr, args.maxnorm, args.batchsize)
  resultdir = os.path.join(outdir,dirstr)
  if not os.path.exists(resultdir):
    os.makedirs(resultdir)

  # Hyperparameters
  lr = args.lr # learning rate
  num_epochs = args.num_epochs
  batch_size = args.batchsize

  # Read train data
  train_files = os.listdir(traindir)
  shape_str = train_files[0].split('.npy')[0].split('_')[2]
  [num_train, seqlen, channels] = [int(elem) for elem in shape_str.split('x')]

  train_samples1 = np.memmap(os.path.join(traindir, 'train_samples1_'+shape_str+'.npy'), dtype='float32',\
                             mode='r', shape=(num_train, seqlen, channels))
  train_samples2 = np.memmap(os.path.join(traindir, 'train_samples2_'+shape_str+'.npy'), dtype='float32',\
                             mode='r', shape=(num_train, seqlen, channels))
  train_labels = np.memmap(os.path.join(traindir, 'train_labels_'+shape_str+'.npy'), dtype='int32',\
                           mode='r', shape=(num_train,))

  # Read validation data
  val_files = os.listdir(valdir)
  shape_str = val_files[0].split('.npy')[0].split('_')[2]
  [num_val, seqlen, channels] = [int(elem) for elem in shape_str.split('x')]

  val_samples1 = np.memmap(os.path.join(valdir, 'val_samples1_'+shape_str+'.npy'), dtype='float32',\
                             mode='r', shape=(num_val, seqlen, channels))
  val_samples2 = np.memmap(os.path.join(valdir, 'val_samples2_'+shape_str+'.npy'), dtype='float32',\
                             mode='r', shape=(num_val, seqlen, channels))
  val_labels = np.memmap(os.path.join(valdir, 'val_labels_'+shape_str+'.npy'), dtype='int32',\
                           mode='r', shape=(num_val,))

  # Read test data
  test_files = os.listdir(testdir)
  shape_str = test_files[0].split('.npy')[0].split('_')[2]
  [num_test, seqlen, channels] = [int(elem) for elem in shape_str.split('x')]

  test_samples1 = np.memmap(os.path.join(testdir, 'test_samples1_'+shape_str+'.npy'), dtype='float32',\
                             mode='r', shape=(num_test, seqlen, channels))
  test_samples2 = np.memmap(os.path.join(testdir, 'test_samples2_'+shape_str+'.npy'), dtype='float32',\
                             mode='r', shape=(num_test, seqlen, channels))
  test_labels = np.memmap(os.path.join(testdir, 'test_labels_'+shape_str+'.npy'), dtype='int32',\
                           mode='r', shape=(num_test,))

  # Data generators for train/val/test
  train_gen = DataGenerator(train_samples1, train_samples2, train_labels,\
                            batch_size=batch_size, seqlen=seqlen, channels=channels,\
                            shuffle=True, augment=False, aug_factor=0.75)
  val_gen = DataGenerator(val_samples1, val_samples2, val_labels,\
                          batch_size=batch_size, seqlen=seqlen, channels=channels)
  test_gen = DataGenerator(test_samples1, test_samples2, test_labels,\
                           batch_size=batch_size, seqlen=seqlen, channels=channels)
  
  # Create model
  fcn_model = FCN(input_shape=(seqlen, channels), norm_max=args.maxnorm)
  samp1 = Input(shape=(seqlen, channels))
  enc_samp1 = fcn_model(samp1)
  samp2 = Input(shape=(seqlen, channels))
  enc_samp2 = fcn_model(samp2)
  diff_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
  diff_enc = diff_layer([enc_samp1, enc_samp2])

  dense_out = Dense(50,activation='relu',
                 kernel_constraint=MaxNorm(args.maxnorm,axis=[0,1]),
                 bias_constraint=MaxNorm(args.maxnorm,axis=0),
                 kernel_initializer=glorot_uniform(seed=0))(diff_enc)
  output = Dense(1,activation='sigmoid',
                 kernel_constraint=MaxNorm(args.maxnorm,axis=[0,1]),
                 bias_constraint=MaxNorm(args.maxnorm,axis=0),
                 kernel_initializer=glorot_uniform(seed=0))(dense_out)
  model = Model(inputs=[samp1,samp2], outputs=output)

  model.compile(optimizer=Adam(lr=lr),
                loss=BinaryCrossentropy(),
                metrics=['accuracy'])

  # Train model
  # Use early stopping and model checkpoints to handle overfitting and save best model
  model_checkpt = ModelCheckpoint(os.path.join(resultdir,'{epoch:02d}-{val_accuracy:.4f}.h5'),\
                                               monitor='val_accuracy',\
                                               mode='max', save_best_only=True)
  batch_renorm_cb = BatchRenormScheduler(len(train_gen)) # Implement batchrenorm after 1st epoch
  history = model.fit(train_gen, epochs=num_epochs, validation_data=val_gen, 
                      verbose=1, shuffle=False,
                      callbacks=[batch_renorm_cb, model_checkpt],
                      workers=2, max_queue_size=20, use_multiprocessing=False)

  # Plot training history
  plot_results(history.history['loss'], history.history['val_loss'],\
               os.path.join(resultdir,'loss.jpg'), metric='Loss')
  plot_results(history.history['accuracy'], history.history['val_accuracy'],\
               os.path.join(resultdir,'accuracy.jpg'), metric='Accuracy')
  
  # Predict probability on validation data using best model
  best_model_file, epoch, val_accuracy = get_best_model(resultdir)
  print('Predicting with model saved at Epoch={:d} with val_accuracy={:0.4f}'.format(epoch, val_accuracy))
  model.load_weights(os.path.join(resultdir,best_model_file))
  probs = model.predict(test_gen)
  y_pred = probs.argmax(axis=1)
  y_true = fold_labels[test_indices]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--traindir', type=str, help='directory with train data')
  parser.add_argument('--valdir', type=str, help='directory with val data')
  parser.add_argument('--testdir', type=str, help='directory with test data')
  parser.add_argument('--outdir', type=str, help='output directory to store results and models')
  parser.add_argument('--lr', type=float, default=0.001, help='learning rate')        
  parser.add_argument('--batchsize', type=int, default=64, help='batch size')        
  parser.add_argument('--maxnorm', type=float, default=1, help='maximum norm for constraint')        
  parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs to run')        
  args = parser.parse_args()
  main(args)
