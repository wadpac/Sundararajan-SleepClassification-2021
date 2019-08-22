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

from metrics import macro_f1
from data_augmentation import jitter, time_warp, rotation, rand_sampling
import matplotlib.pyplot as plt

np.random.seed(2)

import tensorflow as tf
import keras
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))

class F1scoreHistory(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.f1score = {'train':[], 'val':[]}
    self.mean_f1score = {'train':[], 'val':[]}

  def on_batch_end(self, batch, logs={}):
    self.f1score['train'].append(logs.get('macro_f1'))
    self.mean_f1score['train'].append(np.mean(self.f1score['train']))
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

# augment data - aug_factor : factor by which majority class samples are augmented. 
# Samples from all other classes will be augmented by same amount
def augment(X, y, sleep_states, fold, aug_factor=1.0, step_sz = 10000):
  if not os.path.exists('tmp'):
    os.makedirs('tmp')
    
  # List of possible variations
  variant_func = [jitter, time_warp, rotation, rand_sampling]

  y_lbl = y.argmax(axis=1)
  y_lbl = [sleep_states[i] for i in y_lbl]
  y_ctr = Counter(y_lbl).most_common()
  print(y_ctr)

  # Find max number of samples for each class
  # Create memory mapped arrays to store data after augmentation
  max_samp = int(y_ctr[0][1]*aug_factor) 
  naug_samp = max_samp * len(sleep_states)
  X_aug = np.memmap('tmp/X_aug_fold'+str(fold)+'.np', dtype='float32', mode='w+', shape=(naug_samp,X.shape[1],X.shape[2]))
  y_aug = np.memmap('tmp/y_aug_fold'+str(fold)+'.np', dtype='int32', mode='w+', shape=(naug_samp,y.shape[1]))
  for i,state in enumerate(y_ctr):
    lbl,ctr = state
    idx = sleep_states.index(lbl)
    lbl_y = y[y[:,idx] == 1,:]
    lbl_x = X[y[:,idx] == 1,:]
    st_idx = i*max_samp

    # Augment each class with random variations
    # Choose a random set of samples and apply a single variation 
    # Toss a coin and choose to apply another variation
    # Append new samples to augmented data
    n_aug = max_samp - ctr 
    if n_aug > 0:
      print('%s: Augmenting %d samples to %d samples' % (lbl,n_aug,ctr))
      end_idx = st_idx + lbl_x.shape[0]
      X_aug[st_idx:end_idx,:,:] = lbl_x
      y_aug[st_idx:end_idx,:] = lbl_y
      if n_aug <= lbl_x.shape[0]: # Few samples to be augmented
        rand_idx = np.random.randint(0,lbl_x.shape[0],n_aug)
        aug_x = random.choice(variant_func)(lbl_x[rand_idx])
        toss = random.choice([0,1])
        if toss == 1:
          aug_x = random.choice(variant_func)(aug_x)
        st_idx = end_idx
        end_idx = st_idx + aug_x.shape[0]
        X_aug[st_idx:end_idx,:,:] = aug_x
        aug_y = np.zeros((n_aug,len(sleep_states))); aug_y[:,idx] = 1
        y_aug[st_idx:end_idx,:] = aug_y
      else: # If more samples to be augmented, create variations in batches of step_sz
        aug_x = None; aug_y = None
        for j in range(0,n_aug,step_sz):
          if j+step_sz < n_aug:
              sz = step_sz
          else:
              sz = n_aug-j
          rand_idx = np.random.randint(0,lbl_x.shape[0],sz)
          x_var = random.choice(variant_func)(lbl_x[rand_idx])
          toss = random.choice([0,1])
          if toss == 1:
            x_var = random.choice(variant_func)(x_var)
          y_var = np.zeros((sz,len(sleep_states))); y_var[:,idx] = 1
          if aug_x is not None:
            aug_x = np.vstack((aug_x,x_var))
            aug_y = np.vstack((aug_y,y_var))
          else:
            aug_x = x_var
            aug_y = y_var
        st_idx = end_idx
        end_idx = st_idx + aug_x.shape[0]
        X_aug[st_idx:end_idx,:,:] = aug_x
        y_aug[st_idx:end_idx,:] = aug_y
    else:
      print('%s: Choosing %d samples of %d samples' % (lbl,max_samp,ctr))
      end_idx = st_idx + max_samp
      rand_idx = np.random.randint(0,lbl_x.shape[0],max_samp)
      X_aug[st_idx:end_idx,:,:] = lbl_x[rand_idx]
      y_aug[st_idx:end_idx,:] = lbl_y[rand_idx]


  # Shuffle indices
  shuf_idx = np.arange(X_aug.shape[0])
  np.random.shuffle(shuf_idx)
  X_aug = X_aug[shuf_idx]
  y_aug = y_aug[shuf_idx]

  y_lbl = y_aug.argmax(axis=1)
  y_lbl = [sleep_states[i] for i in y_lbl]
  y_ctr = Counter(y_lbl).most_common()
  print(y_ctr)
  
  # Flush and close memmap objects
  del(X_aug); del(y_aug)

  return naug_samp

def limit_mem():
  K.get_session().close()
  cfg = K.tf.ConfigProto()
  cfg.gpu_options.allow_growth = True
  K.set_session(K.tf.Session(config=cfg))

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
    naug_samp = augment(out_X_train, out_y_train, sleep_states, fold=fold, aug_factor=0.5)
    out_X_train = np.memmap('tmp/X_aug_fold'+str(fold)+'.np', dtype='float32', mode='r', \
                            shape=(naug_samp,out_X_train.shape[1],out_X_train.shape[2]))
    out_y_train = np.memmap('tmp/y_aug_fold'+str(fold)+'.np', dtype='int32', mode='r', shape=(naug_samp,out_y_train.shape[1]))
    out_X_test = X[test_indices]; out_y_test = y[test_indices]
    out_lbl = out_y_train.argmax(axis=1)
    out_users = [users[k] for k in test_indices]

    # Normalize data
    scaler = StandardScaler()
    train_nsamp, train_nseq, train_nch = out_X_train.shape
    out_X_train = scaler.fit_transform(out_X_train.reshape(train_nsamp,-1)).reshape(train_nsamp, train_nseq, train_nch)
    test_nsamp, test_nseq, test_nch = out_X_test.shape
    out_X_test = scaler.transform(out_X_test.reshape(test_nsamp,-1)).reshape(test_nsamp, test_nseq, test_nch)

    # Inner CV
    val_acc = []
    models = []
    strat_kfold = StratifiedKFold(n_splits=inner_cv_splits, random_state=0, shuffle=False)
    for grp_train_indices, grp_test_indices in strat_kfold.split(out_X_train, out_lbl):
      grp_train_indices = sample(list(grp_train_indices),len(grp_train_indices))
      in_X_train = out_X_train[grp_train_indices]; in_y_train = out_y_train[grp_train_indices]
      grp_test_indices = sample(list(grp_test_indices),len(grp_test_indices)//3)
      in_X_test = out_X_train[grp_test_indices]; in_y_test = out_y_train[grp_test_indices]
      #print(Counter(in_y_train[:1000].argmax(axis=1))); continue
   
      limit_mem() 
      # Generate candidate architectures
      model = modelgen.generate_models(in_X_train.shape, \
                                    number_of_classes=num_classes, \
                                    number_of_models=1, metrics=[macro_f1])#, model_type='CNN')  

      # Compare generated architectures on a subset of data for few epochs
      outfile = os.path.join(resultdir, 'model_comparison.json')
      hist, acc, loss = find_architecture.train_models_on_samples(in_X_train, \
                                 in_y_train, in_X_test, in_y_test, model, nr_epochs=1, \
                                 subset_size=len(grp_train_indices)//3, verbose=True, batch_size=50, \
                                 outputfile=outfile, metric='macro_f1')
      val_acc.append(acc[0])
      models.append(model[0])

    # Choose best model and evaluate values on validation data
    print('Evaluating on best model for fold %d'% fold)
    best_model_index = np.argmax(val_acc)
    best_model, best_params, best_model_type = models[best_model_index]
    print('Best model type and parameters:')
    print(best_model_type)
    print(best_params)
  
    nr_epochs = 10
    ntrain = out_X_train.shape[0]; nval = ntrain//5
    val_idx = np.random.randint(ntrain, size=nval)
    train_idx = [i for i in range(out_X_train.shape[0]) if i not in val_idx]
    trainX = out_X_train[train_idx]; trainY = out_y_train[train_idx]
    valX = out_X_train[val_idx]; valY = out_y_train[val_idx]
    
    limit_mem()
    if best_model_type == 'CNN':
      best_model = modelgen.generate_CNN_model(trainX.shape, num_classes, filters=best_params['filters'], \
                                      fc_hidden_nodes=best_params['fc_hidden_nodes'], \
                                      learning_rate=best_params['learning_rate'], \
                                      regularization_rate=best_params['regularization_rate'], \
                                      metrics=[macro_f1])
    else:
      best_model = modelgen.generate_DeepConvLSTM_model(trainX.shape, num_classes, filters=best_params['filters'], \
                                      lstm_dims=best_params['lstm_dims'], \
                                      learning_rate=best_params['learning_rate'], \
                                      regularization_rate=best_params['regularization_rate'], \
                                      metrics=[macro_f1])

    # Use early stopping and model checkpoints to handle overfitting and save best model
    model_checkpt = ModelCheckpoint(os.path.join(resultdir,'best_model_fold'+str(fold)+'.h5'), monitor='val_macro_f1',\
                                                 mode='max', save_best_only=True)
    history = F1scoreHistory()
    hist = best_model.fit(trainX, trainY, epochs=nr_epochs, batch_size=50, \
                             validation_data=(valX, valY), callbacks=[early_stopping, model_checkpt, history])

    # Plot training history
    plt.Figure()
    plt.plot(history.mean_f1score['train'])
    #plt.plot(history.mean_f1score['val'])
    plt.title('Model F1-score')
    plt.ylabel('F1-score')
    plt.xlabel('Batch')
    #plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(resultdir,'Fold'+str(fold)+'_performance_curve.jpg'))
    
#    # Save model
#    best_model.save(os.path.join(resultdir,'best_model_fold'+str(fold)+'.h5'))

    # Predict probability on validation data
    probs = best_model.predict_proba(out_X_test, batch_size=1)
    y_pred = probs.argmax(axis=1)
    y_true = out_y_test.argmax(axis=1)
    predictions.append((out_users, y_true, y_pred))

    # Save user report
    if mode == 'binary':
      save_user_report(predictions, sleep_states, os.path.join(resultdir,'fold'+str(fold)+'_deeplearning_binary_results.csv'))
    else:
      save_user_report(predictions, sleep_states, os.path.join(resultdir,'fold'+str(fold)+'_deeplearning_multiclass_results.csv'))
  
  get_classification_report(predictions, sleep_states)

  # Save user report
  if mode == 'binary':
    save_user_report(predictions, sleep_states, os.path.join(resultdir,'deeplearning_binary_results.csv'))
  else:
    save_user_report(predictions, sleep_states, os.path.join(resultdir,'deeplearning_multiclass_results.csv'))

if __name__ == "__main__":
  main(sys.argv[1:])
