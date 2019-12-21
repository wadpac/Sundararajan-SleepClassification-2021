import sys,os
import numpy as np
import pandas as pd
import h5py
import random
from random import sample
import json
import tensorflow as tf
from mcfly import modelgen, find_architecture
from keras.models import load_model
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight

import matplotlib.pyplot as plt

np.random.seed(2)

import tensorflow as tf
import keras
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Dropout, TimeDistributed, BatchNormalization, GlobalAveragePooling1D
from keras.layers import CuDNNLSTM # Comment on HPC
from keras.regularizers import l2
from keras.optimizers import Adam

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim

from data_augmentation import augment
from metrics import macro_f1, get_one_hot, fbeta, macro_precision, macro_recall
from hyperas_models import data, DeepConvLSTM

class F1scoreHistory(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.f1score = {'train':[], 'val':[]}
    self.mean_f1score = {'train':[], 'val':[]}

  def on_batch_end(self, batch, logs={}):
    self.f1score['train'].append(logs.get('macro_f1'))
    self.mean_f1score['train'].append(np.mean(self.f1score['train'][-500:]))
    #self.f1score['val'].append(logs.get('val_macro_f1'))
    #self.mean_f1score['val'].append(np.mean(self.f1score['val'][-100:]))

def save_user_report(pred_list, sleep_states, fname):
  nfolds = len(pred_list)
  for i in range(nfolds):
    users = pred_list[i][0]
    y_true = pred_list[i][1]
    y_true = [sleep_states[idx] for idx in y_true]
    y_pred = pred_list[i][2]
    y_pred = [sleep_states[idx] for idx in y_pred]
    fold = np.array([i+1]*len(users))
    df = pd.DataFrame({'Fold':fold, 'Users':users, 'Y_true':y_true, 'Y_pred':y_pred})
    if i != 0:
      df.to_csv(fname, mode='a', header=False, index=False)
    else:
      df.to_csv(fname, mode='w', header=True, index=False)

def get_classification_report(pred_list, sleep_states):
  nfolds = len(pred_list)
  precision = 0.0; recall = 0.0; fscore = 0.0; accuracy = 0.0
  class_metrics = {}
  for state in sleep_states:
    class_metrics[state] = {'precision':0.0, 'recall': 0.0, 'f1-score':0.0}
  confusion_mat = np.zeros((len(sleep_states),len(sleep_states)))
  for i in range(nfolds):
    y_true = pred_list[i][1]
    y_pred = pred_list[i][2]
    prec, rec, fsc, sup = precision_recall_fscore_support(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    precision += prec; recall += rec; fscore += fsc; accuracy += acc
    fold_class_metrics = classification_report(y_true, y_pred, \
                                          target_names=sleep_states, output_dict=True)
    for state in sleep_states:
      class_metrics[state]['precision'] += fold_class_metrics[state]['precision']
      class_metrics[state]['recall'] += fold_class_metrics[state]['recall']
      class_metrics[state]['f1-score'] += fold_class_metrics[state]['f1-score']

    fold_conf_mat = confusion_matrix(y_true, y_pred).astype(np.float)
    for idx,state in enumerate(sleep_states):
      fold_conf_mat[idx,:] = fold_conf_mat[idx,:] / float(len(y_true[y_true == idx]))
    confusion_mat = confusion_mat + fold_conf_mat

  precision = precision/nfolds; recall = recall/nfolds
  fscore = fscore/nfolds; accuracy = accuracy/nfolds
  print('\nPrecision = %0.4f' % (precision*100.0))
  print('Recall = %0.4f' % (recall*100.0))
  print('F-score = %0.4f' % (fscore*100.0))
  print('Accuracy = %0.4f' % (accuracy*100.0))

  # Classwise report
  print('\nClass\t\tPrecision\tRecall\t\tF1-score')
  for state in sleep_states:
    class_metrics[state]['precision'] = class_metrics[state]['precision'] / nfolds
    class_metrics[state]['recall'] = class_metrics[state]['recall'] / nfolds
    class_metrics[state]['f1-score'] = class_metrics[state]['f1-score'] / nfolds
    print('%s\t\t%0.4f\t\t%0.4f\t\t%0.4f' % (state, class_metrics[state]['precision'], \
                      class_metrics[state]['recall'], class_metrics[state]['f1-score']))
  print('\n')

  # Confusion matrix
  confusion_mat = confusion_mat / nfolds
  if len(sleep_states) > 2:
    print('ConfMat\tWake\tNREM1\tNREM2\tNREM3\tREM\n')
    for i in range(confusion_mat.shape[0]):
      print('%s\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f' % (sleep_states[i], confusion_mat[i][0], confusion_mat[i][1], confusion_mat[i][2], confusion_mat[i][3], confusion_mat[i][4]))
    print('\n')
  else:
    print('ConfMat\tWake\tSleep\n')
    for i in range(confusion_mat.shape[0]):
      print('%s\t%0.4f\t%0.4f' % (sleep_states[i], confusion_mat[i][0], confusion_mat[i][1]))
    print('\n')

def main(argv):
  infile = argv[0]
  mode = argv[1] # binary or multiclass
  outdir = argv[2]

  if mode == 'multiclass':
    sleep_states = ['Wake', 'NREM 1', 'NREM 2', 'NREM 3', 'REM']
  else:
    sleep_states = ['Wake', 'Sleep']

  if not os.path.exists(outdir):
    os.makedirs(outdir)

  resultdir = os.path.join(outdir,mode,'models')
  if not os.path.exists(resultdir):
    os.makedirs(resultdir)

  all_data = np.load(infile)
  X = all_data['data']
  y = all_data['labels']
  if mode == 'binary':
    y = np.array([y[:,0], y[:,1:].any(axis=-1)]).T
  users = all_data['user']
  dataset = all_data['dataset']
  #X = X[dataset == 'UPenn']
  #y = y[dataset == 'UPenn']
  num_classes = y.shape[1]
 
  # Shuffle data
  shuf_idx = np.arange(X.shape[0])
  np.random.shuffle(shuf_idx)
  X = X[shuf_idx]
  y = y[shuf_idx]
  users = [users[i] for i in shuf_idx]

  # Get small subset
  #idx = np.random.randint(X.shape[0],size=10000)
  #X = X[idx]; y = y[idx]; users = [users[i] for i in idx]
  y_lbl = y.argmax(axis=1)
  y_lbl = [sleep_states[i] for i in y_lbl]

  early_stopping = EarlyStopping(monitor='val_macro_f1', mode='max', verbose=1, patience=2)
 
  # Use nested cross-validation based on users
  # Outer CV
  outer_cv_splits = 5; inner_cv_splits = 5
  group_kfold = GroupKFold(n_splits=outer_cv_splits)
  fold = 0
  predictions = []
  for train_indices, test_indices in group_kfold.split(X,y,users):
    fold += 1
    print('Evaluating fold %d' % fold)
    out_X_train = X[train_indices]; out_y_train = y[train_indices]
    out_lbl = out_y_train.argmax(axis=1)
    out_class_wts = class_weight.compute_class_weight('balanced', np.unique(out_lbl), out_lbl) # Compute class weights before augmentation
    out_X_test = X[test_indices]; out_y_test = y[test_indices]
    out_lbl = out_y_train.argmax(axis=1)
    out_users = [users[k] for k in test_indices]
    naug_samp = augment(out_X_train, out_y_train, sleep_states, aug_factor=1.25)
    
    # Normalize data
    out_X_train = np.memmap('tmp/X_aug.np', dtype='float32', mode='r+', shape=(naug_samp, out_X_train.shape[1], out_X_train.shape[2]))
    scaler = StandardScaler()
    train_nsamp, train_nseq, train_nch = out_X_train.shape
    out_X_train = scaler.fit_transform(out_X_train.reshape(train_nsamp,-1)).reshape(train_nsamp, train_nseq, train_nch)
    del(out_X_train) # flush out train data after scaling to write back to disk
    test_nsamp, test_nseq, test_nch = out_X_test.shape
    out_X_test = scaler.transform(out_X_test.reshape(test_nsamp,-1)).reshape(test_nsamp, test_nseq, test_nch)

    data_info = {'nsamp': train_nsamp, 'seqlen': train_nseq, 'nchannel': train_nch,\
                 'nclass': out_y_train.shape[1], 'class_wt': list(out_class_wts)}
    with open('tmp/data_info.csv','w') as fp:
      fp.write(json.dumps(data_info))

    # Run hyperparam optimization with hyperas
    best_run, best_model = optim.minimize(model=DeepConvLSTM, data=data, algo=tpe.suggest,\
                                          max_evals=5, trials=Trials(), functions=[macro_f1, get_one_hot, fbeta, macro_precision, macro_recall],\
                                          keep_temp=True)
    print(best_run)


#    # Choose best model and evaluate values on validation data
#    print('Evaluating on best model for fold %d'% fold)
#    best_model_index = np.argmax(val_acc)
#    best_model, best_params, best_model_type = models[best_model_index]
#    print('Best model type and parameters:')
#    print(best_model_type)
#    print(best_params)
#  
#    nr_epochs = 10
#    ntrain = out_X_train.shape[0]; nval = ntrain//5
#    val_idx = np.random.randint(ntrain, size=nval)
#    train_idx = [i for i in range(out_X_train.shape[0]) if i not in val_idx]
#    trainX = out_X_train[train_idx]; trainY = out_y_train[train_idx]
#    valX = out_X_train[val_idx]; valY = out_y_train[val_idx]
#    
#    limit_mem()
#    if best_model_type == 'CNN':
#      best_model = modelgen.generate_CNN_model(trainX.shape, num_classes, filters=best_params['filters'], \
#                                      fc_hidden_nodes=best_params['fc_hidden_nodes'], \
#                                      learning_rate=best_params['learning_rate'], \
#                                      regularization_rate=best_params['regularization_rate'], \
#                                      metrics=[macro_f1])
#    else:
#      best_model = modelgen.generate_DeepConvLSTM_model(trainX.shape, num_classes, filters=best_params['filters'], \
#                                      lstm_dims=best_params['lstm_dims'], \
#                                      learning_rate=best_params['learning_rate'], \
#                                      regularization_rate=best_params['regularization_rate'], \
#                                      metrics=[macro_f1])
#
#    # Use early stopping and model checkpoints to handle overfitting and save best model
#    model_checkpt = ModelCheckpoint(os.path.join(resultdir,'best_model_fold'+str(fold)+'.h5'), monitor='val_macro_f1',\
#                                                 mode='max', save_best_only=True)
#    history = F1scoreHistory()
#    hist = best_model.fit(trainX, trainY, epochs=nr_epochs, batch_size=50, \
#                             validation_data=(valX, valY), class_weight=out_class_wts, callbacks=[early_stopping, model_checkpt, history])
#
#    # Plot training history
#    plt.Figure()
#    plt.plot(history.mean_f1score['train'])
#    #plt.plot(history.mean_f1score['val'])
#    plt.title('Model F1-score')
#    plt.ylabel('F1-score')
#    plt.xlabel('Batch')
#    #plt.legend(['Train', 'Test'], loc='upper left')
#    plt.savefig(os.path.join(resultdir,'Fold'+str(fold)+'_performance_curve.jpg'))
#    plt.clf()
#    
##    # Save model
##    best_model.save(os.path.join(resultdir,'best_model_fold'+str(fold)+'.h5'))
#
#    # Predict probability on validation data
#    probs = best_model.predict_proba(out_X_test, batch_size=1)
#    y_pred = probs.argmax(axis=1)
#    y_true = out_y_test.argmax(axis=1)
#    predictions.append((out_users, y_true, y_pred))
#
#    # Save user report
#    if mode == 'binary':
#      save_user_report(predictions, sleep_states, os.path.join(resultdir,'fold'+str(fold)+'_deeplearning_binary_results.csv'))
#    else:
#      save_user_report(predictions, sleep_states, os.path.join(resultdir,'fold'+str(fold)+'_deeplearning_multiclass_results.csv'))
#  
#  get_classification_report(predictions, sleep_states)
#
#  # Save user report
#  if mode == 'binary':
#    save_user_report(predictions, sleep_states, os.path.join(resultdir,'deeplearning_binary_results.csv'))
#  else:
#    save_user_report(predictions, sleep_states, os.path.join(resultdir,'deeplearning_multiclass_results.csv'))
#
if __name__ == "__main__":
  main(sys.argv[1:])
