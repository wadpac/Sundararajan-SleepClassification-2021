import sys,os
import numpy as np
import pandas as pd
import random
import argparse
from collections import Counter
from tqdm import tqdm
import h5py
import shutil

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

import kerastuner
from kerastuner.tuners import Hyperband

from hypermodel import ResnetHyperModel
from resnet import Resnet
from tuner import CVTuner
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

  resultdir = os.path.join(outdir,mode,'models')
  if not os.path.exists(resultdir):
    os.makedirs(resultdir)

  # Load pretrained model
  pretrained_model = load_model(os.path.join(modeldir, 'pretrained_model.h5'))
  pretrained_renet_weights = None
  for layer in pretrained_model.layers:
    if layer.name == "model":
      pretrained_resnet_weights = layer.get_weights()

  # Hyperparameters
  num_epochs = args.num_epochs
  num_channels = args.num_channels # number of raw data channels
  feat_channels = args.feat_channels # Add ENMO, z-angle and LIDS as additional channels
  batchsize = args.batchsize # Batchsize
  hp_epochs = args.hp_epochs # No. of hyperparameter validation epochs
  lr = args.lr # Learning rate
  batchsize = args.batchsize # Batch size

  resultdir = os.path.join(outdir,mode,'lr-{:4f}_batchsize-{:d}'.format(lr, batchsize))
  if not os.path.exists(resultdir):
    os.makedirs(resultdir)

  model_hyperparam = {}
  model_hyperparam['maxnorm'] = [0.5, 1.0, 2.0, 3.0]
  model_hyperparam['dense_units'] = {'min': 100, 'max': 700, 'step': 50}
  model_hyperparam['dropout'] = [0.1, 0.2, 0.3]
  model_hyperparam['lr'] = [1e-3, 1e-4]

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

  num_classes = len(states)

  # Get valid values for CV split 
  valid_indices = data[labels != -1].index.values
  
  # dummy values for partition as raw data cannot be loaded to memory
  X = data[['ENMO_mean', 'ENMO_std', 'ENMO_mad']].values[valid_indices] 
  y = labels[valid_indices]
  groups = users[valid_indices]

  # Read raw data
  fp = h5py.File(os.path.join(indir, 'all_train_rawdata_30.0s.h5'), 'r')
  raw_data = fp['data']
  [num_samples, seqlen, n_channels] = raw_data.shape

  # Read raw data statistics
  stats = np.load(os.path.join(modeldir, "stats.npz"))
  mean = stats['mean']
  std = stats['std']

  # Use nested cross-validation based on users
  # Outer CV
  outer_cv_splits = 5; inner_cv_splits = 5
  out_fold = 0; predictions = []
  outer_group_kfold = GroupKFold(n_splits=outer_cv_splits)
  for train_indices, test_indices in outer_group_kfold.split(X, y, groups):
    if os.path.exists('untitled_project'):
      shutil.rmtree('untitled_project')
    out_fold += 1  
    print('Evaluating outer fold %d' % (out_fold))
    out_fold_train_indices = valid_indices[train_indices]; out_fold_test_indices = valid_indices[test_indices]
    out_fold_y_train = labels[out_fold_train_indices]; out_fold_y_test = labels[out_fold_test_indices]
    out_fold_users_train = users[out_fold_train_indices]; out_fold_users_test = users[out_fold_test_indices]
    out_fold_ts_test = ts[out_fold_test_indices]
    out_fold_fnames_test = fnames[out_fold_test_indices]

    # Build a hypermodel for hyperparam search
    hyperModel = ResnetHyperModel(hyperparam=model_hyperparam, seqlen=seqlen,\
                                  channels=num_channels+feat_channels,\
                                  pretrained_wts=pretrained_resnet_weights,\
                                  num_classes=num_classes)   
    tuner = CVTuner(hypermodel=hyperModel,
                    oracle=kerastuner.oracles.Hyperband(objective='val_loss', max_epochs=3),
                    cv=inner_cv_splits, states=states, num_classes=num_classes,
                    seqlen=seqlen, num_channels=num_channels, feat_channels=feat_channels,
                    mean=mean, std=std)
    # Use a subset of training data for hyperparam search
#    train_users = list(set(out_fold_users_train))
#    random.shuffle(train_users)
#    nsubtrain_users = int(0.5*len(train_users))
#    sub_train_users = np.array(train_users[:nsubtrain_users])
#    sub_train_indices = out_fold_train_indices[np.isin(out_fold_users_train, sub_train_users)]
    tuner.search(data=raw_data, labels=labels, users=users, indices=out_fold_train_indices, batch_size=batchsize)

    # Train fold with best best hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]
    with open(os.path.join(resultdir, 'fold'+str(out_fold)+'_hyperparameters.txt'),"w") as fp:
        fp.write("Maxnorm = {:.2f}\n".format(best_hp.values['maxnorm']))
        fp.write("Learning rate = {:.4f}\n".format(best_hp.values['lr']))
        fp.write("Preclassification layer units = {:d}\n".format(best_hp.values['preclassification']))
        fp.write("Dropout = {:.2f}\n".format(best_hp.values['dropout']))
    print(best_hp.values)
    model = tuner.hypermodel.build(best_hp)
    for layer in model.layers:
      if layer.name == "model":
        layer.set_weights(pretrained_resnet_weights)
    print(model.summary())

    # Data generators
    # Split train data into train and validation data based on users
    trainval_users = list(set(out_fold_users_train))
    random.shuffle(trainval_users)
    ntrainval_users = len(trainval_users)
    nval_users = int(0.1*ntrainval_users); ntrain_users = ntrainval_users - nval_users
    train_users = np.array(trainval_users[:ntrain_users]); val_users = np.array(trainval_users[ntrain_users:])
    out_fold_val_indices = out_fold_train_indices[np.isin(out_fold_users_train, val_users)]
    out_fold_train_indices = out_fold_train_indices[np.isin(out_fold_users_train, train_users)]

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

    # Use callback to compute F-scores over entire validation data
    metrics_cb = Metrics(val_data=val_gen, batch_size=batchsize)
    # Use early stopping and model checkpoints to handle overfitting and save best model
    model_checkpt = ModelCheckpoint(os.path.join(resultdir,'fold'+str(out_fold)+'_'+mode+'-{epoch:02d}-{val_f1:.4f}.h5'),\
                                                 monitor='val_f1',\
                                                 mode='max', save_best_only=True)
    batch_renorm_cb = BatchRenormScheduler(len(train_gen))
    history = model.fit(train_gen, epochs=num_epochs, validation_data=val_gen, 
                        verbose=1, shuffle=False,
                        callbacks=[batch_renorm_cb, metrics_cb, model_checkpt],
                        workers=2, max_queue_size=20, use_multiprocessing=False)

    # Plot training history
    plot_results(out_fold+1, history.history['loss'], history.history['val_loss'],\
                 os.path.join(resultdir,'Fold'+str(out_fold)+'_'+mode+'_loss.jpg'), metric='Loss')
    plot_results(out_fold+1, history.history['accuracy'], history.history['val_accuracy'],\
                 os.path.join(resultdir,'Fold'+str(out_fold)+'_'+mode+'_accuracy.jpg'), metric='Accuracy')
    plot_results(out_fold+1, history.history['macro_f1'], metrics_cb.val_f1,\
                 os.path.join(resultdir,'Fold'+str(out_fold)+'_'+mode+'_macro_f1.jpg'), metric='Macro F1')
    
    # Predict probability on validation data using best model
    best_model_file, epoch, val_f1 = get_best_model(resultdir, out_fold)
    print('Predicting with model saved at Epoch={:d} with val_f1={:0.4f}'.format(epoch, val_f1))
    model.load_weights(os.path.join(resultdir,best_model_file))
    probs = model.predict(test_gen)
    y_pred = probs.argmax(axis=1)
    y_true = out_fold_y_test
    predictions.append((users[test_indices], data.iloc[test_indices]['timestamp'], 
                        data.iloc[test_indices]['filename'], test_indices, y_true, probs))

    # Save user report
    cv_save_classification_result(predictions, states, 
                                  os.path.join(resultdir,'fold'+str(out_fold)+'_deeplearning_' + mode + '_results.csv'), method='dl')
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
  parser.add_argument('--hp_iter', type=int, default=10, help='#hyperparameter iterations')        
  parser.add_argument('--hp_epochs', type=int, default=2, help='#hyperparam validation epochs')        
  parser.add_argument('--batchsize', type=int, default=64, help='batch size range')        
  args = parser.parse_args()
  main(args)
