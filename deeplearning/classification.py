import sys,os
import numpy as np
import pandas as pd
import random
import argparse
from collections import Counter
from tqdm import tqdm
import h5py

import tensorflow as tf
from tensorflow.keras.models import load_model
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

def get_best_model(indir, fold, mode='max'):
  files = os.listdir(indir)
  files = [fname for fname in files if fname.startswith('fold'+str(fold)) and fname.endswith('.h5')]
  metric = np.array([float(fname.split('.h5')[0].split('-')[2]) for fname in files])
  epoch = np.array([int(fname.split('.h5')[0].split('-')[1]) for fname in files])
  best_idx = np.argmax(metric) if mode == 'max' else np.argmin(metric)
  return files[best_idx], epoch[best_idx], metric[best_idx]

def main(argv):
  indir = args.indir
  mode = args.mode # binary or multiclass or nonwear
  modeldir = args.modeldir
  outdir = args.outdir

  if mode == 'multiclass':
    states = ['Wake', 'NREM 1', 'NREM 2', 'NREM 3', 'REM']
  elif mode == 'binary':
    states = ['Wake', 'Sleep', 'Wake_ext']
    collate_states = ['NREM 1', 'NREM 2', 'NREM 3', 'REM']
  elif mode == 'nonwear':
    states = ['Wear', 'Nonwear']
    collate_states = ['Wake', 'NREM 1', 'NREM 2', 'NREM 3', 'REM']

  num_classes = len(states) 

  if not os.path.exists(outdir):
    os.makedirs(outdir)

  resultdir = os.path.join(outdir,mode,'models')
  if not os.path.exists(resultdir):
    os.makedirs(resultdir)

  # Hyperparameters
  num_epochs = args.num_epochs
  num_channels = args.num_channels # number of raw data channels
  feat_channels = args.feat_channels # Add ENMO, z-angle and LIDS as additional channels
  hp_iter = args.hp_iter # No. of hyperparameter iterations
  hp_epochs = args.hp_epochs # No. of hyperparameter validation epochs

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
  labels = np.array([states.index(i) if i in states else -1 for i in labels])

  # Get valid values for CV split 
  valid_indices = data[data['label'].isin(states)].index.values
  # dummy values for partition as raw data cannot be loaded to memory
  X = data[['ENMO_mean', 'ENMO_std', 'ENMO_mad']].values[valid_indices] 
  y = labels[valid_indices]
  groups = users[valid_indices]
 
  # Read raw data
  fp = h5py.File(os.path.join(indir, 'all_train_rawdata_30.0s.h5'), 'r')
  raw_data = fp['data']
  [num_samples, seqlen, n_channels] = raw_data.shape

  # Use nested cross-validation based on users
  # Outer CV
  outer_cv_splits = 5; inner_cv_splits = 5
  out_fold = 0; predictions = []
  outer_group_kfold = GroupKFold(n_splits=outer_cv_splits)
  for train_indices, test_indices in outer_group_kfold.split(X, y, groups):
    out_fold += 1  
    print('Evaluating fold %d' % (out_fold))
    out_fold_train_indices = valid_indices[train_indices]; out_fold_test_indices = valid_indices[test_indices]
    out_fold_y_train = labels[out_fold_train_indices]; out_fold_y_test = labels[out_fold_test_indices]
    out_fold_users_train = users[out_fold_train_indices]; out_fold_users_test = users[out_fold_test_indices]
    out_fold_ts_test = ts[out_fold_test_indices]
    out_fold_fnames_test = fnames[out_fold_test_indices]

    class_wts = compute_class_weight(class_weight='balanced', classes=np.unique(out_fold_y_train),
                                     y=out_fold_y_train)

    # Hyperparameter selection
    grp_X_train = X[train_indices]; grp_y_train = y[train_indices]; grp_users_train = groups[train_indices]
    for hp in range(1):#hp_iter):
      lr = 10 ** (-np.random.randint(args.lr_low, args.lr_high+1)) 
      maxnorm = np.random.randint(args.maxnorm_low, args.maxnorm_high+1) 
      batchsize = np.random.randint(args.batchsize_low, args.batchsize_high+1)
      batchsize = batchsize - batchsize%8 # make batchsize multiple of 8

      # Inner CV 
      inner_group_kfold = GroupKFold(n_splits=inner_cv_splits)
      for grp_train_idx, grp_val_idx in \
          inner_group_kfold.split(grp_X_train, grp_y_train, grp_users_train):
        in_fold_train_indices = valid_indices[train_indices[grp_train_idx]]
        in_fold_val_indices = valid_indices[train_indices[grp_val_idx]]

        train_gen = DataGenerator(in_fold_train_indices, raw_data, labels, states, partition='train',\
                              batch_size=batchsize, seqlen=seqlen, n_channels=num_channels, feat_channels=feat_channels,\
                              n_classes=num_classes, shuffle=True, augment=True, aug_factor=0.75, balance=True,
                              mean=mean, std=std)
        val_gen = DataGenerator(in_fold_val_indices, raw_data, labels, states, partition='val',\
                            batch_size=batchsize, seqlen=seqlen, n_channels=num_channels, feat_channels=feat_channels,\
                            n_classes=num_classes, mean=mean, std=std)

#    # Create model
#    # Use batchnorm as first step since computing mean and std 
#    # across entire dataset is time-consuming
#    model = FCN(input_shape=(seqlen,num_channels+feat_channels), max_seqlen=max_seqlen,
#                num_classes=len(states), norm_max=args.maxnorm)
#    #print(model.summary()); exit()
#    model.compile(optimizer=Adam(lr=lr),
#                  loss=focal_loss(),
#                  metrics=['accuracy', macro_f1])
#
#    # Train model
#    # Use callback to compute F-scores over entire validation data
#    metrics_cb = Metrics(val_data=val_gen, batch_size=batch_size)
#    # Use early stopping and model checkpoints to handle overfitting and save best model
#    model_checkpt = ModelCheckpoint(os.path.join(resultdir,'fold'+str(fold+1)+'_'+mode+'-{epoch:02d}-{val_f1:.4f}.h5'),\
#                                                 monitor='val_f1',\
#                                                 mode='max', save_best_only=True)
#    batch_renorm_cb = BatchRenormScheduler(len(train_gen))
#    history = model.fit(train_gen, epochs=num_epochs, validation_data=val_gen, 
#                        verbose=1, shuffle=False,
#                        callbacks=[batch_renorm_cb, metrics_cb, model_checkpt],
#                        workers=2, max_queue_size=20, use_multiprocessing=False)
#
#    # Plot training history
#    plot_results(fold+1, history.history['loss'], history.history['val_loss'],\
#                 os.path.join(resultdir,'Fold'+str(fold+1)+'_'+mode+'_loss.jpg'), metric='Loss')
#    plot_results(fold+1, history.history['accuracy'], history.history['val_accuracy'],\
#                 os.path.join(resultdir,'Fold'+str(fold+1)+'_'+mode+'_accuracy.jpg'), metric='Accuracy')
#    plot_results(fold+1, history.history['macro_f1'], metrics_cb.val_f1,\
#                 os.path.join(resultdir,'Fold'+str(fold+1)+'_'+mode+'_macro_f1.jpg'), metric='Macro F1')
#    
#    # Predict probability on validation data using best model
#    best_model_file, epoch, val_f1 = get_best_model(resultdir, fold+1)
#    print('Predicting with model saved at Epoch={:d} with val_f1={:0.4f}'.format(epoch, val_f1))
#    model.load_weights(os.path.join(resultdir,best_model_file))
#    probs = model.predict(test_gen)
#    y_pred = probs.argmax(axis=1)
#    y_true = fold_labels[test_indices]
#    predictions.append((users[test_indices], data.iloc[test_indices]['timestamp'], 
#                        data.iloc[test_indices]['filename'], test_indices, y_true, probs))
#
#    # Save user report
#    cv_save_classification_result(predictions, states, 
#                                  os.path.join(resultdir,'fold'+str(fold+1)+'_deeplearning_' + mode + '_results.csv'), method='dl')
#    cv_get_classification_report(predictions, mode, states, method='dl')
#  
#  cv_get_classification_report(predictions, mode, states, method='dl')
#
#  # Save user report
#  cv_save_classification_result(predictions, states,
#                                os.path.join(resultdir,'deeplearning_' + mode + '_results.csv'), method='dl')

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
  parser.add_argument('--hp_iter', type=int, default=10, help='#hyperparameter iterations')        
  parser.add_argument('--hp_epochs', type=int, default=2, help='#hyperparam validation epochs')        
  parser.add_argument('--lr_low', type=int, default=1, help='learning rate range - lower (log scale)')        
  parser.add_argument('--lr_high', type=int, default=4, help='learning rate range - upper (log scale)')        
  parser.add_argument('--batchsize_low', type=int, default=8, help='batch size range - lower')        
  parser.add_argument('--batchsize_high', type=int, default=64, help='batch size range - upper')        
  parser.add_argument('--maxnorm_low', type=float, default=1, help='maximum norm range - lower')        
  parser.add_argument('--maxnorm_high', type=float, default=20, help='maximum norm range - upper')        
  args = parser.parse_args()
  main(args)
