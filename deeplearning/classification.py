import sys,os
import numpy as np
import pandas as pd
import random
import argparse
from collections import Counter

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight

import matplotlib.pyplot as plt

from FCN import FCN
from datagenerator import DataGenerator
from transforms import get_LIDS
from metrics import macro_f1
from callbacks import Metrics, BatchRenormScheduler
from losses import focal_loss

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
  plt.ylim(0,1)
  plt.legend(['Train', 'Val'], loc='upper right')
  plt.savefig(out_fname)
  plt.clf()

def get_partition(data, labels, users, sel_users, sleep_states, is_train=False):
  wake_idx = sleep_states.index('Wake')
  wake_ext_idx = sleep_states.index('Wake_ext')
  indices = np.arange(len(users))[np.isin(users, sel_users)] 
  if is_train: # use extra wake samples only for training
    # Determine LIDS score of wake samples  
    wake_indices = indices[labels[indices] == wake_idx]
    wake_samp = data[wake_indices,:,5].mean(axis=1) # LIDS mean
    wake_perc = np.percentile(wake_samp,50)
    # Choose extra wake samples whose LIDS score is less than 50% percentile of LIDS score of wake samples
    wake_ext_indices = indices[labels[indices] == wake_ext_idx]
    wake_ext_samp = data[wake_ext_indices,:,5].mean(axis=1) # LIDS mean
    valid_indices = wake_ext_indices[wake_ext_samp < wake_perc]
    indices = np.concatenate((indices[labels[indices] != wake_ext_idx], np.array(valid_indices)))
  else:
    indices = indices[labels[indices] != wake_ext_idx]
  
  return indices

def main(argv):
  indir = args.indir
  mode = args.mode # binary or multiclass
  outdir = args.outdir

  if mode == 'multiclass':
    sleep_states = ['Wake', 'NREM 1', 'NREM 2', 'NREM 3', 'REM', 'Nonwear', 'Wake_ext']
  else:
    sleep_states = ['Wake', 'Sleep', 'Nonwear', 'Wake_ext']
    collate_sleep = ['NREM 1', 'NREM 2', 'NREM 3', 'REM']

  valid_sleep_states = [state for state in sleep_states if state != 'Wake_ext']
  num_classes = len(valid_sleep_states) 

  if not os.path.exists(outdir):
    os.makedirs(outdir)

  resultdir = os.path.join(outdir,mode,'models')
  if not os.path.exists(resultdir):
    os.makedirs(resultdir)

  # Read data from disk
  data = pd.read_csv(os.path.join(indir,'features_30.0s.csv'))
  labels = data['label'].values
  users = data['user'].values
  if mode == 'binary':
    labels = ['Sleep' if lbl in collate_sleep else lbl for lbl in labels]

  # Read raw data
  shape_df = pd.read_csv(os.path.join(indir,'datashape_30.0s.csv'))
  num_samples = shape_df['num_samples'].values[0]
  seqlen = shape_df['num_timesteps'].values[0]
  n_channels = shape_df['num_channels'].values[0]
  raw_data = np.memmap(os.path.join(indir,'rawdata_30.0s.npz'), dtype='float32', mode='r', shape=(num_samples, seqlen, n_channels))

  #early_stopping = EarlyStopping(monitor='val_macro_f1', mode='max', verbose=1, patience=2)

  # Hyperparameters
  lr = args.lr # learning rate
  num_epochs = args.num_epochs
  batch_size = args.batchsize
  max_seqlen = 1504
  num_channels = args.num_channels # number of raw data channels
  feat_channels = args.feat_channels # Add ENMO, z-angle and LIDS as additional channels

  # Use nested cross-validation based on users
  # Outer CV
  unique_users = list(set(users))
  random.shuffle(unique_users)
  cv_splits = 5
  fold_nusers = len(unique_users) // cv_splits
  if len(unique_users) % cv_splits:
    fold_nusers += 1
  predictions = []
  wake_idx = sleep_states.index('Wake')
  wake_ext_idx = sleep_states.index('Wake_ext')
  for fold in range(cv_splits):
    print('Evaluating fold %d' % (fold+1))
    test_users = unique_users[fold*fold_nusers:(fold+1)*fold_nusers]
    trainval_users = [user for user in unique_users if user not in test_users] 
    train_users = trainval_users[:int(0.8*len(trainval_users))]
    val_users = trainval_users[len(train_users):]

    # Create partitions
    # make a copy to change wake_ext for this fold 
    fold_labels = np.array([sleep_states.index(lbl) for lbl in labels])
    train_indices = get_partition(raw_data, fold_labels, users, train_users, sleep_states, is_train=True)
    val_indices = get_partition(raw_data, fold_labels, users, val_users, sleep_states)
    test_indices = get_partition(raw_data, fold_labels, users, test_users, sleep_states)
    nsamples = len(train_indices) + len(val_indices) + len(test_indices)
    print('Train: {:0.2f}%, Val: {:0.2f}%, Test: {:0.2f}%'\
            .format(len(train_indices)*100.0/nsamples, len(val_indices)*100.0/nsamples,\
                    len(test_indices)*100.0/nsamples))
  
    # Rename wake_ext as wake for training samples
    rename_indices = train_indices[fold_labels[train_indices] == wake_ext_idx]
    fold_labels[rename_indices] = wake_idx
    
    # Data generators for computing statistics
    stat_gen = DataGenerator(train_indices, raw_data, fold_labels, valid_sleep_states, partition='stat',\
                              batch_size=batch_size, seqlen=seqlen, n_channels=num_channels, feat_channels=feat_channels,\
                              n_classes=num_classes, shuffle=True)
    mean, std = stat_gen.fit()
    np.savez(os.path.join(resultdir,'Fold'+str(fold+1)+'_stats'), mean=mean, std=std)
    
    # Data generators for train/val/test
    train_gen = DataGenerator(train_indices, raw_data, fold_labels, valid_sleep_states, partition='train',\
                              batch_size=batch_size, seqlen=seqlen, n_channels=num_channels, feat_channels=feat_channels,\
                              n_classes=num_classes, shuffle=True, augment=True, aug_factor=0.75, balance=True,
                              mean=mean, std=std)
    val_gen = DataGenerator(val_indices, raw_data, fold_labels, valid_sleep_states, partition='val',\
                            batch_size=batch_size, seqlen=seqlen, n_channels=num_channels, feat_channels=feat_channels,\
                            n_classes=num_classes, mean=mean, std=std)
    test_gen = DataGenerator(test_indices, raw_data, fold_labels, valid_sleep_states, partition='test',\
                             batch_size=batch_size, seqlen=seqlen, n_channels=num_channels, feat_channels=feat_channels,\
                             n_classes=num_classes, mean=mean, std=std)

    # Create model
    # Use batchnorm as first step since computing mean and std 
    # across entire dataset is time-consuming
    model = FCN(input_shape=(seqlen,num_channels+feat_channels), max_seqlen=max_seqlen,
                num_classes=len(valid_sleep_states))
    #print(model.summary())
    model.compile(optimizer=Adam(lr=lr), loss=focal_loss(),
                  metrics=['accuracy', macro_f1])

    # Train model
    # Use callback to compute F-scores over entire validation data
    metrics_cb = Metrics(val_data=val_gen, batch_size=batch_size)
    # Use early stopping and model checkpoints to handle overfitting and save best model
    model_checkpt = ModelCheckpoint(os.path.join(resultdir,'best_model_fold'+str(fold+1)+'.h5'),\
                                                 monitor='val_f1',\
                                                 mode='max', save_best_only=True)
    batch_renorm_cb = BatchRenormScheduler(len(train_gen))
    history = model.fit(train_gen, epochs=num_epochs, validation_data=val_gen, 
                                  verbose=1, shuffle=False, callbacks=[batch_renorm_cb, metrics_cb, model_checkpt],
                                  workers=2, max_queue_size=20, use_multiprocessing=False)
 
    # Plot training history
    plot_results(fold+1, history.history['loss'], history.history['val_loss'],\
                 os.path.join(resultdir,'Fold'+str(fold+1)+'_loss.jpg'), metric='Loss')
    plot_results(fold+1, history.history['accuracy'], history.history['val_accuracy'],\
                 os.path.join(resultdir,'Fold'+str(fold+1)+'_accuracy.jpg'), metric='Accuracy')
    plot_results(fold+1, history.history['macro_f1'], metrics_cb.val_f1,\
                 os.path.join(resultdir,'Fold'+str(fold+1)+'_macro_f1.jpg'), metric='Macro F1')
    
    # Predict probability on validation data using best model
    model.load_weights(os.path.join(resultdir,'best_model_fold'+str(fold+1)+'.h5'))
    probs = model.predict(test_gen)
    y_pred = probs.argmax(axis=1)
    y_true = fold_labels[test_indices]
    predictions.append((users[test_indices], data.iloc[test_indices]['timestamp'], 
                        data.iloc[test_indices]['filename'], test_indices, y_true, probs))

    # Save user report
    if mode == 'binary':
      cv_save_classification_result(predictions, valid_sleep_states, 
                                    os.path.join(resultdir,'fold'+str(fold+1)+'_deeplearning_binary_results.csv'), method='dl')
    else:
      cv_save_classification_result(predictions, valid_sleep_states, 
                                    os.path.join(resultdir,'fold'+str(fold+1)+'_deeplearning_multiclass_results.csv'), method='dl')
    cv_get_classification_report(predictions, mode, valid_sleep_states, method='dl')
  
  cv_get_classification_report(predictions, mode, valid_sleep_states, method='dl')

  # Save user report
  if mode == 'binary':
    cv_save_classification_result(predictions, valid_sleep_states,
                                  os.path.join(resultdir,'deeplearning_binary_results.csv'), method='dl')
  else:
    cv_save_classification_result(predictions, valid_sleep_states,
                                  os.path.join(resultdir,'deeplearning_multiclass_results.csv'), method='dl')

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--indir', type=str, help='input directory containing data and labels')
  parser.add_argument('--mode', type=str, default='binary', help='classification mode - binary/multiclass')
  parser.add_argument('--outdir', type=str, help='output directory to store results and models')
  parser.add_argument('--lr', type=float, default=0.001, help='learning rate')        
  parser.add_argument('--batchsize', type=int, default=64, help='batch size')        
  parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs to run')        
  parser.add_argument('--num_channels', type=int, default=3, help='number of data channels')
  parser.add_argument('--feat_channels', type=int, default=0, help='number of feature channels')
  args = parser.parse_args()
  main(args)
