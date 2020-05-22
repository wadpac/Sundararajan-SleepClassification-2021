import sys,os
import numpy as np
import pandas as pd
import random
import argparse
from collections import Counter
from tqdm import tqdm
import h5py

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Lambda, Dropout
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt

from resnet import Resnet
from datagenerator import DataGenerator
from transforms import get_LIDS
from metrics import macro_f1
from callbacks import Metrics, BatchRenormScheduler
from losses import focal_loss, weighted_categorical_crossentropy, train_val_loss

sys.path.append('../analysis/')
from analysis import cv_save_classification_result, cv_get_classification_report

np.random.seed(2)

# Limit GPU memory allocated
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  tf.config.experimental.set_memory_growth(gpus[0], True)

def plot_results(fold, train_result, val_result, out_fname, metric='Loss'):
  plt.Figure()
  plt.plot(train_result, label='train')
  plt.plot(val_result, label='val')
  plt.title('{} for fold {}'.format(metric, fold))
  plt.ylabel(metric)
  plt.xlabel('Epochs')
  ylim = 1.0 #if metric != 'Loss' else 5.0
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

def create_resnet_model(seqlen, num_channels, maxnorm, modeldir, dense_units, num_classes):
  resnet_model = Resnet(input_shape=(seqlen,num_channels), norm_max=maxnorm)
  # Load weights from pretrained model
  resnet_model.load_weights(os.path.join(modeldir, 'pretrained_resnet.h5'))
  samp = Input(shape=(seqlen, num_channels))
  enc_samp = resnet_model(samp)
  dense_out = Dense(dense_units, activation='relu',
                 kernel_constraint=MaxNorm(maxnorm,axis=[0,1]),
                 bias_constraint=MaxNorm(maxnorm,axis=0),
                 kernel_initializer=glorot_uniform(seed=0), name='FC1')(enc_samp)
  dense_out = Dropout(rate=0.2)(dense_out)
  output = Dense(num_classes, activation='softmax',
                 kernel_constraint=MaxNorm(maxnorm,axis=[0,1]),
                 bias_constraint=MaxNorm(maxnorm,axis=0),
                 kernel_initializer=glorot_uniform(seed=0), name='output')(dense_out)
  model = Model(inputs=samp, outputs=output)
  return model

def main(argv):
  indir = args.indir
  mode = args.mode # binary or multiclass or nonwear
  modeldir = args.modeldir
  outdir = args.outdir

  if mode == 'multiclass':
    states = ['Wake', 'NREM 1', 'NREM 2', 'NREM 3', 'REM']
  elif mode == 'binary':
    states = ['Wake', 'Sleep']
    collate_states = ['NREM 1', 'NREM 2', 'NREM 3', 'REM']
  elif mode == 'nonwear':
    states = ['Wear', 'Nonwear']
    collate_states = ['Wake', 'NREM 1', 'NREM 2', 'NREM 3', 'REM']

  num_classes = len(states) 

  if not os.path.exists(outdir):
    os.makedirs(outdir)

  # Hyperparameters
  num_epochs = args.num_epochs
  num_channels = args.num_channels # number of raw data channels
  feat_channels = args.feat_channels # Add ENMO, z-angle and LIDS as additional channels
  hp_iter = args.hp_iter # No. of hyperparameter iterations
  hp_epochs = args.hp_epochs # No. of hyperparameter validation epochs
  lr = args.lr # Learning rate
  batchsize = args.batchsize # Batch size

  resultdir = os.path.join(outdir,mode,'lr-{:4f}_batchsize-{:d}'.format(lr, batchsize))
  if not os.path.exists(resultdir):
    os.makedirs(resultdir)

  # Read data from disk
  data = pd.read_csv(os.path.join(indir,'all_train_features_30.0s.csv'))
  ts = data['timestamp']
  fnames = data['filename']
  labels = data['label']
  users = data['user'].astype(str).values
  if mode == 'binary':
    labels = np.array(['Sleep' if lbl in collate_states else lbl for lbl in labels])
  elif mode == 'nonwear':
    labels = np.array(['Wear' if lbl in collate_states else lbl for lbl in labels])

#  unique_users = list(set(users))
#  random.shuffle(unique_users)
#  unique_users = unique_users[:10]

  # Get valid values for CV split 
  #valid_indices = data[(np.isin(labels, states)) & (np.isin(users, unique_users))].index.values
  valid_indices = data[np.isin(labels, states)].index.values
  labels = np.array([states.index(i) if i in states else -1 for i in labels])

  # dummy values for partition as raw data cannot be loaded to memory
  X = data[['ENMO_mean', 'ENMO_std', 'ENMO_mad']].values[valid_indices] 
  y = labels[valid_indices]
  groups = users[valid_indices]
 
  # Read raw data
  fp = h5py.File(os.path.join(indir, 'all_train_rawdata_30.0s.h5'), 'r')
  raw_data = fp['data']
  [num_samples, seqlen, n_channels] = raw_data.shape

  # Get statistics
  stats = np.load(os.path.join(modeldir,'stats.npz'))
  mean = stats['mean']; std = stats['std']

  # Use nested cross-validation based on users
  # Outer CV
  outer_cv_splits = 5; inner_cv_splits = 5
  out_fold = 0; predictions = []
  outer_group_kfold = GroupKFold(n_splits=outer_cv_splits)
  for train_indices, test_indices in outer_group_kfold.split(X, y, groups):
    out_fold += 1  
    print('Evaluating fold %d' % (out_fold))
    out_fold_train_indices = valid_indices[train_indices]
    out_fold_y_train = labels[out_fold_train_indices]
    out_fold_users_train = users[out_fold_train_indices]

    class_wts = compute_class_weight(class_weight='balanced', classes=np.unique(out_fold_y_train),
                                     y=out_fold_y_train)

    # Hyperparameter selection
    grp_X_train = X[train_indices]; grp_y_train = y[train_indices]; grp_users_train = groups[train_indices]
    best_val_f1 = 0
    for hp in range(hp_iter):
      dense_units = np.random.randint(args.dense_low, args.dense_high+1)
      dense_units = dense_units - dense_units % 10
      maxnorm = np.random.randint(args.maxnorm_low, args.maxnorm_high+1) 

      # Inner CV
      in_fold = 0; inner_cv_val_f1 = 0
      inner_group_kfold = GroupKFold(n_splits=inner_cv_splits)
      for grp_train_idx, grp_val_idx in \
          inner_group_kfold.split(grp_X_train, grp_y_train, grp_users_train):
        in_fold += 1      
        in_fold_train_indices = valid_indices[train_indices[grp_train_idx]]
        in_fold_val_indices = valid_indices[train_indices[grp_val_idx]]

        train_gen = DataGenerator(in_fold_train_indices, raw_data, labels, states, partition='train',\
                              batch_size=batchsize, seqlen=seqlen, n_channels=num_channels, feat_channels=feat_channels,\
                              n_classes=num_classes, shuffle=True, augment=True, aug_factor=0.75, balance=True,
                              mean=mean, std=std)
        val_gen = DataGenerator(in_fold_val_indices, raw_data, labels, states, partition='val',\
                            batch_size=batchsize, seqlen=seqlen, n_channels=num_channels, feat_channels=feat_channels,\
                            n_classes=num_classes, mean=mean, std=std)
         
        # Create model
        model = create_resnet_model(seqlen, num_channels+feat_channels, maxnorm,\
                                    modeldir, dense_units, num_classes)
        model.compile(optimizer=Adam(lr=lr),
                      loss=focal_loss(),
                      metrics=['accuracy', macro_f1])

        # Train model
        # Use callback to compute F-scores over entire validation data
        metrics_cb = Metrics(val_data=val_gen, batch_size=batchsize)
        history = model.fit(train_gen, epochs=hp_epochs, validation_data=val_gen, validation_freq=hp_epochs, 
                            verbose=1, shuffle=False, steps_per_epoch=len(val_gen)*2,
                            callbacks=[metrics_cb],
                            workers=2, max_queue_size=20, use_multiprocessing=False)
        inner_cv_val_f1 += metrics_cb.val_f1[-1]
        print('HPIter {:d} - Inner CV {:d} : val_f1 = {:4f}'.format(hp, in_fold, metrics_cb.val_f1[-1]))
      
      inner_cv_val_f1 = inner_cv_val_f1 / float(inner_cv_splits)
      print('HPIter {:d}: val_f1 = {:4f}'.format(hp, inner_cv_val_f1))
      if inner_cv_val_f1 > best_val_f1:
        best_val_f1 = inner_cv_val_f1
        best_dense_units = dense_units
        best_maxnorm = maxnorm
        print('Best so far: Dense = {:d}, maxnorm = {:d}, val_f1 = {:4f}'.format(best_dense_units, best_maxnorm, best_val_f1))
   
    # Use best hyperparameters to train the model for out_fold
    print('Best: Dense = {:d}, maxnorm = {:d}, val_f1 = {:4f}'.format(best_dense_units, best_maxnorm, best_val_f1))
    fold_train_users = users[valid_indices[train_indices]]
    fold_unique_users = list(set(fold_train_users))
    random.shuffle(fold_unique_users)
    num_val_users = max(2, len(fold_unique_users)//5)
    fold_train_users = fold_unique_users[:-num_val_users]; fold_val_users = fold_unique_users[-num_val_users:]
    
    out_fold_train_indices = valid_indices[train_indices[np.isin(groups[train_indices], fold_train_users)]]
    out_fold_y_train = labels[out_fold_train_indices]
    out_fold_users_train = users[out_fold_train_indices]

    out_fold_val_indices = valid_indices[train_indices[np.isin(groups[train_indices], fold_val_users)]]
    out_fold_y_val = labels[out_fold_val_indices]
    out_fold_users_val = users[out_fold_val_indices]
    
    out_fold_test_indices = valid_indices[test_indices]
    out_fold_y_test = labels[out_fold_test_indices]
    out_fold_users_test = users[out_fold_test_indices]
    out_fold_ts_test = ts[out_fold_test_indices]
    out_fold_fnames_test = fnames[out_fold_test_indices]

    # Data generators for outer CV
    train_gen = DataGenerator(out_fold_train_indices, raw_data, labels, states, partition='train',\
                          batch_size=batchsize, seqlen=seqlen, n_channels=num_channels, feat_channels=feat_channels,\
                          n_classes=num_classes, shuffle=True, augment=True, aug_factor=0.75, balance=True,
                          mean=mean, std=std)
    val_gen = DataGenerator(out_fold_val_indices, raw_data, labels, states, partition='val',\
                        batch_size=batchsize, seqlen=seqlen, n_channels=num_channels, feat_channels=feat_channels,\
                        n_classes=num_classes, mean=mean, std=std)
    test_gen = DataGenerator(out_fold_test_indices, raw_data, labels, states, partition='test',\
                        batch_size=batchsize, seqlen=seqlen, n_channels=num_channels, feat_channels=feat_channels,\
                        n_classes=num_classes, mean=mean, std=std)
     
    # Create model for outer CV
    model = create_resnet_model(seqlen, num_channels+feat_channels, best_maxnorm,\
                                modeldir, best_dense_units, num_classes)
    model.compile(optimizer=Adam(lr=lr),
                  loss=focal_loss(),
                  metrics=['accuracy', macro_f1])

    # Train model
    # Use callback to compute F-scores over entire validation data
    metrics_cb = Metrics(val_data=val_gen, batch_size=batchsize)
    #batch_renorm_cb = BatchRenormScheduler(len(train_gen)) # Implement batchrenorm after 1st epoch
    # Use early stopping and model checkpoints to handle overfitting and save best model
    folddir = os.path.join(resultdir, 'fold{:d}_dense-{:d}_maxnorm-{:4f}'.format(out_fold, best_dense_units, best_maxnorm))
    if not os.path.exists(folddir):
      os.makedirs(folddir)
    model_checkpt = ModelCheckpoint(os.path.join(folddir, '{epoch:02d}-{val_f1:.4f}.h5'),\
                                                 monitor='val_f1')#,\
                                                 #mode='max', save_best_only=True)
    history = model.fit(train_gen, epochs=num_epochs, validation_data=val_gen, verbose=1, shuffle=False,
                        callbacks=[metrics_cb, model_checkpt],
                        workers=2, max_queue_size=20, use_multiprocessing=False)

    # Plot training history
    plot_results(out_fold, history.history['loss'], history.history['val_loss'],\
                 os.path.join(folddir,'loss.jpg'), metric='Loss')
    plot_results(out_fold, history.history['accuracy'], history.history['val_accuracy'],\
                 os.path.join(folddir,'accuracy.jpg'), metric='Accuracy')
    plot_results(out_fold, history.history['macro_f1'], metrics_cb.val_f1,\
                 os.path.join(folddir,'macro_f1.jpg'), metric='Macro F1')
    
    # Predict probability on validation data using best model
    best_model_file, epoch, val_f1 = get_best_model(folddir)
    print('Predicting with model saved at Epoch={:d} with val_f1={:0.4f}'.format(epoch, val_f1))
    model.load_weights(os.path.join(folddir,best_model_file))
    probs = model.predict(test_gen)
    y_pred = probs.argmax(axis=1)
    y_true = out_fold_y_test
    predictions.append((out_fold_users_test, out_fold_ts_test, 
                        out_fold_fnames_test, out_fold_test_indices, y_true, probs))
 
    # Save user report
    cv_save_classification_result(predictions, states, 
                                  os.path.join(modeldir,'fold'+str(out_fold)+'_deeplearning_' + mode + '_results.csv'), method='dl')
    cv_get_classification_report(predictions, mode, states, method='dl')
  
  cv_get_classification_report(predictions, mode, states, method='dl')

  # Save user report
  cv_save_classification_result(predictions, states,
                                os.path.join(resultdir,'deeplearning_' + mode + '_results.csv'), method='dl')

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--indir', type=str, help='input directory containing data and labels')
  parser.add_argument('--mode', type=str, default='binary', help='classification mode - binary/multiclass')
  parser.add_argument('--modeldir', type=str, help='directory with pretrained models and normalization info')
  parser.add_argument('--outdir', type=str, help='output directory to store results and models')
  parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs to run')        
  parser.add_argument('--num_channels', type=int, default=3, help='number of data channels')
  parser.add_argument('--feat_channels', type=int, default=0, help='number of feature channels')
  # Hyperparameter selection
  parser.add_argument('--hp_iter', type=int, default=5, help='#hyperparameter iterations')        
  parser.add_argument('--hp_epochs', type=int, default=1, help='#hyperparam validation epochs')        
  parser.add_argument('--lr', type=float, default=0.001, help='learning rate')        
  parser.add_argument('--batchsize', type=int, default=32, help='batch size')        
  parser.add_argument('--dense_low', type=int, default=30, help='#dense units - lower')        
  parser.add_argument('--dense_high', type=int, default=200, help='#dense units - upper')        
  parser.add_argument('--maxnorm_low', type=float, default=1, help='maximum norm range - lower')        
  parser.add_argument('--maxnorm_high', type=float, default=10, help='maximum norm range - upper')        
  args = parser.parse_args()
  main(args)
