import sys,os
import numpy as np
import pandas as pd
import h5py
import random
from random import sample
import tensorflow as tf
from mcfly import modelgen, find_architecture
from keras.models import load_model
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight

from metrics import macro_f1
from data_augmentation import augment, load_as_memmap
import matplotlib.pyplot as plt

np.random.seed(2)

import tensorflow as tf
import keras
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#keras.backend.set_session(tf.Session(config=config))

from mcfly_datagenerator import DataGenerator

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

def get_partition(files, labels, users, sel_users, sleep_states, is_train=False):
  wake_idx = sleep_states.index('Wake')
  wake_ext_idx = sleep_states.index('Wake_ext')
  labels = np.array([sleep_states.index(lbl) for lbl in labels])
  if is_train: # use extra wake samples only for training
    indices = np.array([i for i,user in enumerate(users) if (user in sel_users)])
  else: 
    indices = np.array([i for i,user in enumerate(users) if ((user in sel_users) and (labels[i] != wake_ext_idx))])
  part_files = np.array(files)[indices]
  part_labels = labels[indices]
  if is_train: # relabel extra wake samples as wake for training
    part_labels[part_labels == wake_ext_idx] = wake_idx
  return part_files, part_labels

def main(argv):
  indir = argv[0]
  mode = argv[1] # binary or multiclass
  outdir = argv[2]

  if mode == 'multiclass':
    sleep_states = ['Wake', 'NREM 1', 'NREM 2', 'NREM 3', 'REM', 'Wake_ext']
  else:
    sleep_states = ['Wake', 'Sleep', 'Wake_ext']
    collate_sleep = ['NREM 1', 'NREM 2', 'NREM 3', 'REM']

  valid_sleep_states = [state for state in sleep_states if state != 'Wake_ext']
  num_classes = len(valid_sleep_states) 

  if not os.path.exists(outdir):
    os.makedirs(outdir)

  resultdir = os.path.join(outdir,mode,'models')
  if not os.path.exists(resultdir):
    os.makedirs(resultdir)

  # Read data from disk
  data = pd.read_csv(os.path.join(indir,'labels.txt'), sep='\t')
  files = []; labels = []; users = []
  for idx, row in data.iterrows():
    files.append(os.path.join(indir, row['filename']) + '.npy')
    labels.append(row['labels'])
    users.append(row['user'])
  if mode == 'binary':
    labels = ['Sleep' if lbl in collate_sleep else lbl for lbl in labels]

  early_stopping = EarlyStopping(monitor='val_macro_f1', mode='max', verbose=1, patience=2)

  seqlen, n_channels = np.load(files[0]).shape
  batch_size = 32
 
  # Use nested cross-validation based on users
  # Outer CV
  unique_users = list(set(users))
  random.shuffle(unique_users)
  out_cv_splits = 5; in_cv_splits = 5
  out_fold_nusers = len(unique_users) // out_cv_splits
  out_n_epochs = 10; in_n_epochs = 1
  predictions = []
  wake_idx = sleep_states.index('Wake')
  wake_ext_idx = sleep_states.index('Wake_ext')
  for out_fold in range(out_cv_splits):
    print('Evaluating fold %d' % (out_fold+1))
    test_users = unique_users[out_fold*out_fold_nusers:(out_fold+1)*out_fold_nusers]
    trainval_users = [user for user in unique_users if user not in test_users] 
    train_users = trainval_users[:int(0.8*len(trainval_users))]
    val_users = trainval_users[len(train_users):]

    train_fnames, train_labels = get_partition(files, labels, users, train_users, sleep_states, is_train=True)
    val_fnames, val_labels = get_partition(files, labels, users, val_users, sleep_states)
    test_fnames, test_labels = get_partition(files, labels, users, test_users, sleep_states)
    
    out_train_gen = DataGenerator(train_fnames, train_labels, valid_sleep_states, partition='out_train',\
                                    batch_size=batch_size, seqlen=seqlen, n_channels=n_channels,\
                                    n_classes=num_classes, shuffle=True, augment=False, balance=True)
    print('Fold {}: Computing mean and standard deviation'.format(out_fold+1))
    #mean, std = out_train_gen.fit()
    mean = None; std = None
    out_val_gen = DataGenerator(val_fnames, val_labels, valid_sleep_states, partition='out_val',\
                                  batch_size=1, seqlen=seqlen, n_channels=n_channels,\
                                  n_classes=num_classes, mean=mean, std=std)
    out_test_gen = DataGenerator(test_fnames, test_labels, valid_sleep_states, partition='out_test',\
                                   batch_size=1, seqlen=seqlen, n_channels=n_channels,\
                                   n_classes=num_classes, mean=mean, std=std)

    # Get class weights
    out_class_wts = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
    print(out_class_wts)

    # Inner CV
    val_acc = []
    models = []
    in_fold_nusers = len(trainval_users) // in_cv_splits
    for in_fold in range(in_cv_splits):
      in_val_users = trainval_users[in_fold*in_fold_nusers:(in_fold+1)*in_fold_nusers]
      in_train_users = [user for user in trainval_users if user not in in_val_users] 
   
      in_train_fnames, in_train_labels = get_partition(files, labels, users, in_train_users,\
                                                       sleep_states, is_train=True)
      in_val_fnames, in_val_labels = get_partition(files, labels, users, in_val_users, sleep_states)
    
      in_train_gen = DataGenerator(in_train_fnames, in_train_labels, valid_sleep_states, partition='in_train',\
                                    batch_size=batch_size, seqlen=seqlen, n_channels=n_channels,\
                                    n_classes=num_classes, shuffle=True, augment=False, balance=True, mean=mean, std=std)
      in_val_gen = DataGenerator(in_val_fnames, in_val_labels, valid_sleep_states, partition='in_val',\
                                  batch_size=1, seqlen=seqlen, n_channels=n_channels,\
                                  n_classes=num_classes, mean=mean, std=std)
      
      # Generate candidate architectures
      model = modelgen.generate_models((None, seqlen, n_channels), \
                                    number_of_classes=num_classes, \
                                    number_of_models=1, metrics=[macro_f1])#, model_type='CNN')  

      # Compare generated architectures on a subset of data for few epochs
      outfile = os.path.join(resultdir, 'model_comparison.json')
      hist, acc, loss = find_architecture.train_models_on_samples(in_train_gen, in_val_gen,
                                 model, nr_epochs=n_in_epochs, class_weight=out_class_wts, \
                                 verbose=True, outputfile=outfile, metric='macro_f1')
      val_acc.append(acc[0])
      models.append(model[0])
      print(data.columns); exit()

    # Choose best model and evaluate values on validation data
    print('Evaluating on best model for fold %d'% fold)
    best_model_index = np.argmax(val_acc)
    best_model, best_params, best_model_type = models[best_model_index]
    print('Best model type and parameters:')
    print(best_model_type)
    print(best_params)
  
    if best_model_type == 'CNN':
      best_model = modelgen.generate_CNN_model((None, seqlen, n_channels), num_classes, filters=best_params['filters'], \
                                      fc_hidden_nodes=best_params['fc_hidden_nodes'], \
                                      learning_rate=best_params['learning_rate'], \
                                      regularization_rate=best_params['regularization_rate'], \
                                      metrics=[macro_f1])
    else:
      best_model = modelgen.generate_DeepConvLSTM_model((None, seqlen, n_channels), num_classes,\
                                      filters=best_params['filters'], \
                                      lstm_dims=best_params['lstm_dims'], \
                                      learning_rate=best_params['learning_rate'], \
                                      regularization_rate=best_params['regularization_rate'], \
                                      metrics=[macro_f1])

    # Use early stopping and model checkpoints to handle overfitting and save best model
    model_checkpt = ModelCheckpoint(os.path.join(resultdir,'best_model_fold'+str(fold)+'.h5'), monitor='val_macro_f1',\
                                                 mode='max', save_best_only=True)
    history = F1scoreHistory()
    hist = best_model.fit_generator(out_train_gen, epochs=n_out_epochs, \
                             validation_data=out_val_gen, class_weight=out_class_wts,\
                             callbacks=[early_stopping, model_checkpt])

    # Plot training history
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

    # Predict probability on validation data
    probs = best_model.predict_generator(out_test_gen)
    y_pred = probs.argmax(axis=1)
    y_true = out_y_test.argmax(axis=1)
    predictions.append((out_users, y_true, y_pred))

    # Save user report
    if mode == 'binary':
      save_user_report(predictions, valid_sleep_states, os.path.join(resultdir,'fold'+str(fold)+'_deeplearning_binary_results.csv'))
    else:
      save_user_report(predictions, valid_sleep_states, os.path.join(resultdir,'fold'+str(fold)+'_deeplearning_multiclass_results.csv'))
  
  get_classification_report(predictions, valid_sleep_states)

  # Save user report
  if mode == 'binary':
    save_user_report(predictions, valid_sleep_states, os.path.join(resultdir,'deeplearning_binary_results.csv'))
  else:
    save_user_report(predictions, valid_sleep_states, os.path.join(resultdir,'deeplearning_multiclass_results.csv'))

if __name__ == "__main__":
  main(sys.argv[1:])
