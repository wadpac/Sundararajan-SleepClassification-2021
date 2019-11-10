import sys,os
import numpy as np
import pandas as pd
import random
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
from losses import weighted_categorical_crossentropy, focal_loss

from tqdm import tqdm

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

def get_classification_report(pred_list, mode, sleep_states):
  nfolds = len(pred_list)
  precision = 0.0; recall = 0.0; fscore = 0.0; accuracy = 0.0
  class_metrics = {}
  sleep_labels = [idx for idx,state in enumerate(sleep_states)]
  for state in sleep_states:
    class_metrics[state] = {'precision':0.0, 'recall': 0.0, 'f1-score':0.0}
  confusion_mat = np.zeros((len(sleep_states),len(sleep_states)))
  for i in range(nfolds):
    y_true = pred_list[i][1]
    y_pred = pred_list[i][2]
    # Get metrics across all classes
    prec, rec, fsc, sup = precision_recall_fscore_support(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    precision += prec; recall += rec; fscore += fsc; accuracy += acc
    # Get metrics per class
    fold_class_metrics = classification_report(y_true, y_pred, labels=sleep_labels,
                                          target_names=sleep_states, output_dict=True)
    for state in sleep_states:
      class_metrics[state]['precision'] += fold_class_metrics[state]['precision']
      class_metrics[state]['recall'] += fold_class_metrics[state]['recall']
      class_metrics[state]['f1-score'] += fold_class_metrics[state]['f1-score']
    # Get confusion matrix
    fold_conf_mat = confusion_matrix(y_true, y_pred, labels=sleep_labels).astype(np.float)
    for idx in range(len(sleep_states)):
      fold_conf_mat[idx,:] = fold_conf_mat[idx,:] / float(len(y_true[y_true == idx]))
    confusion_mat = confusion_mat + fold_conf_mat

  # Average metrics across all folds
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
  if mode == 'multiclass':
    print('ConfMat\tWake\tNREM1\tNREM2\tNREM3\tREM\tNonwear\n')
    for i in range(len(sleep_states)):
      print('%s\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f' % (sleep_states[i], confusion_mat[i][0],\
              confusion_mat[i][1], confusion_mat[i][2], confusion_mat[i][3], confusion_mat[i][4]))
    print('\n')
  else:
    print('ConfMat\tWake\tSleep\tNonwear\n')
    for i in range(len(sleep_states)):
      print('%s\t%0.4f\t%0.4f\t%0.4f' % (sleep_states[i], confusion_mat[i][0],\
              confusion_mat[i][1], confusion_mat[i][2]))
    print('\n')

def get_partition(files, labels, users, sel_users, sleep_states, is_train=False):
  wake_idx = sleep_states.index('Wake')
  wake_ext_idx = sleep_states.index('Wake_ext')
  labels = np.array([sleep_states.index(lbl) for lbl in labels])
  indices = np.arange(len(users))[np.isin(users, sel_users)] #([i for i,user in enumerate(users) if user in sel_users])
  if is_train: # use extra wake samples only for training
    # Determine LIDS score of wake samples  
    wake_indices = indices[labels[indices] == wake_idx]
    wake_samp = np.zeros((len(wake_indices), 1))
    for i,index in enumerate(wake_indices):
      samp = np.load(files[index])
      lids = get_LIDS(samp[:,0], samp[:,1], samp[:,2])
      wake_samp[i] = lids.mean()
    wake_perc = np.percentile(wake_samp,50)
    # Choose extra wake samples whose LIDS score is less than 50% percentile of LIDS score of wake samples
    wake_ext_indices = indices[labels[indices] == wake_ext_idx]
    valid_indices = []
    for i,index in enumerate(wake_ext_indices):
      samp = np.load(files[index])
      lids = get_LIDS(samp[:,0], samp[:,1], samp[:,2])
      if lids.mean() < wake_perc:
        valid_indices.append(index)
    indices = np.concatenate((indices[labels[indices] != wake_ext_idx], np.array(valid_indices)))
  else:
    indices = indices[labels[indices] != wake_ext_idx]
  part_files = np.array(files)[indices]
  part_labels = labels[indices]
  part_users = [users[i] for i in indices]
  if is_train: # relabel extra wake samples as wake for training
    part_labels[part_labels == wake_ext_idx] = wake_idx
  
  return part_files, part_labels, part_users

def main(argv):
  indir = argv[0]
  mode = argv[1] # binary or multiclass
  outdir = argv[2]

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
  print(seqlen, n_channels)

  # Hyperparameters
  lr = 0.0005 # learning rate
  num_epochs = 30
  batch_size = 128
  max_seqlen = 1504
  feat_channels = 3 # Add ENMO, z-angle and LIDS as additional channels

  # Use nested cross-validation based on users
  # Outer CV
  unique_users = list(set(users))
  random.shuffle(unique_users)
  cv_splits = 5
  fold_nusers = len(unique_users) // cv_splits
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
    train_fnames, train_labels, train_users = get_partition(files, labels, users, train_users,\
                                                            sleep_states, is_train=True)
    val_fnames, val_labels, val_users = get_partition(files, labels, users, val_users, sleep_states)
    test_fnames, test_labels, test_users = get_partition(files, labels, users, test_users, sleep_states)
    nsamples = len(train_fnames) + len(val_fnames) + len(test_fnames)
    print('Train: {:0.2f}%, Val: {:0.2f}%, Test: {:0.2f}%'\
            .format(len(train_fnames)*100.0/nsamples, len(val_fnames)*100.0/nsamples,\
                    len(test_fnames)*100.0/nsamples))
    
    # Create data generators 
    train_gen = DataGenerator(train_fnames, train_labels, valid_sleep_states, partition='train',\
                              batch_size=batch_size, seqlen=seqlen, n_channels=n_channels, feat_channels=feat_channels,\
                              n_classes=num_classes, shuffle=True, augment=True, aug_factor=0.75, balance=True)
    val_gen = DataGenerator(val_fnames, val_labels, valid_sleep_states, partition='val',\
                            batch_size=batch_size, seqlen=seqlen, n_channels=n_channels, feat_channels=feat_channels,\
                            n_classes=num_classes)
    test_gen = DataGenerator(test_fnames, test_labels, valid_sleep_states, partition='test',\
                             batch_size=batch_size, seqlen=seqlen, n_channels=n_channels, feat_channels=feat_channels,\
                             n_classes=num_classes)

    # Get class weights
    class_wts = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
   
    # Create model
    # Use batchnorm as first step since computing mean and std 
    # across entire dataset is time-consuming
    model = FCN(input_shape=(seqlen,n_channels+feat_channels), max_seqlen=max_seqlen, num_classes=len(valid_sleep_states))
    print(model.summary())
    model.compile(optimizer=Adam(lr=lr), loss=focal_loss(),
                  metrics=['accuracy', macro_f1])

    # Train model
    # Use early stopping and model checkpoints to handle overfitting and save best model
    model_checkpt = ModelCheckpoint(os.path.join(resultdir,'best_model_fold'+str(fold+1)+'.h5'),\
                                                 monitor='val_macro_f1',\
                                                 mode='max', save_best_only=True)
    history = model.fit(train_gen, epochs=num_epochs, validation_data=val_gen,
                                  verbose=1, shuffle=False, #class_weight=class_wts, #steps_per_epoch=1000,
                                  callbacks=[model_checkpt], workers=2, max_queue_size=100, use_multiprocessing=True)

    # Plot training history
    plot_results(fold+1, history.history['loss'], history.history['val_loss'],\
                 os.path.join(resultdir,'Fold'+str(fold+1)+'_loss.jpg'), metric='Loss')
    plot_results(fold+1, history.history['accuracy'], history.history['val_accuracy'],\
                 os.path.join(resultdir,'Fold'+str(fold+1)+'_accuracy.jpg'), metric='Accuracy')
    plot_results(fold+1, history.history['macro_f1'], history.history['val_macro_f1'],\
                 os.path.join(resultdir,'Fold'+str(fold+1)+'_macro_f1.jpg'), metric='Macro F1')
    
    # Predict probability on validation data using best model
    model.load_weights(os.path.join(resultdir,'best_model_fold'+str(fold+1)+'.h5'))
    probs = model.predict(test_gen)
    y_pred = probs.argmax(axis=1)
    y_true = test_labels
    predictions.append((test_users, y_true, y_pred))

    # Save user report
    if mode == 'binary':
      save_user_report(predictions, valid_sleep_states, os.path.join(resultdir,'fold'+str(fold+1)+'_deeplearning_binary_results.csv'))
    else:
      save_user_report(predictions, valid_sleep_states, os.path.join(resultdir,'fold'+str(fold+1)+'_deeplearning_multiclass_results.csv'))
    get_classification_report(predictions, mode, valid_sleep_states)
  
  get_classification_report(predictions, mode, valid_sleep_states)

  # Save user report
  if mode == 'binary':
    save_user_report(predictions, valid_sleep_states, os.path.join(resultdir,'deeplearning_binary_results.csv'))
  else:
    save_user_report(predictions, valid_sleep_states, os.path.join(resultdir,'deeplearning_multiclass_results.csv'))

if __name__ == "__main__":
  main(sys.argv[1:])
