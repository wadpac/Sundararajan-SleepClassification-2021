import sys,os
import numpy as np
import pandas as pd
import random
from mcfly import modelgen, find_architecture
from keras.models import load_model
from collections import Counter

from sklearn.model_selection import GroupKFold, StratifiedKFold

from metrics import weighted_f1
from data_augmentation import jitter, time_warp, rotation, rand_sampling

np.random.seed(2)

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
    end_idx = st_idx + lbl_x.shape[0]
    X_aug[st_idx:end_idx,:,:] = lbl_x
    y_aug[st_idx:end_idx,:] = lbl_y

    # Augment each class with random variations
    # Choose a random set of samples and apply a single variation 
    # Toss a coin and choose to apply another variation
    # Append new samples to augmented data
    n_aug = max_samp - ctr 
    print('%s: Augmenting %d samples to %d samples' % (lbl,n_aug,ctr))
    if n_aug > 0:
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

def main(argv):
  infile = argv[0]
  outdir = argv[1]

  sleep_states = ['Wake', 'NREM 1', 'NREM 2', 'NREM 3', 'REM']
  if not os.path.exists(outdir):
    os.makedirs(outdir)

  resultdir = os.path.join(outdir,'models')
  if not os.path.exists(resultdir):
    os.makedirs(resultdir)

  all_data = np.load(infile)
  X = all_data['data']
  y = all_data['labels']
  users = all_data['user']
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
 
  # Use nested cross-validation based on users
  # Outer CV
  outer_cv_splits = 5; inner_cv_splits = 3
  group_kfold = GroupKFold(n_splits=outer_cv_splits)
  fold = 0
  for train_indices, test_indices in group_kfold.split(X,y,users):
    fold += 1
    print('Evaluating fold %d' % fold)
    out_X_train = X[train_indices]; out_y_train = y[train_indices]
    naug_samp = augment(out_X_train, out_y_train, sleep_states, fold=fold, aug_factor=1.25)
    out_X_train = np.memmap('tmp/X_aug_fold'+str(fold)+'.np', dtype='float32', mode='r', \
                            shape=(naug_samp,out_X_train.shape[1],out_X_train.shape[2]))
    out_y_train = np.memmap('tmp/y_aug_fold'+str(fold)+'.np', dtype='int32', mode='r', shape=(naug_samp,out_y_train.shape[1]))
    out_X_test = X[test_indices]; out_y_test = y[test_indices]
    out_lbl = out_y_train.argmax(axis=1)

    # Inner CV
    val_acc = []
    models = []
    strat_kfold = StratifiedKFold(n_splits=inner_cv_splits, random_state=0, shuffle=False)
    for grp_train_indices, grp_test_indices in strat_kfold.split(out_X_train, out_lbl):
      in_X_train = out_X_train[grp_train_indices]; in_y_train = out_y_train[grp_train_indices]
      in_X_test = out_X_train[grp_test_indices]; in_y_test = out_y_train[grp_test_indices]
    
      # Generate candidate architectures
      model = modelgen.generate_models(in_X_train.shape, \
                                    number_of_classes=num_classes, \
                                    number_of_models=1, metrics=[weighted_f1])  

      # Compare generated architectures on a subset of data for few epochs
      outfile = os.path.join(resultdir, 'model_comparison.json')
      hist, acc, loss = find_architecture.train_models_on_samples(in_X_train, \
                                 in_y_train, in_X_test, in_y_test, model, nr_epochs=5, \
                                 subset_size=in_X_train.shape[0], verbose=True, \
                                 outputfile=outfile, metric='weighted_f1')
      val_acc.append(acc[0])
      models.append(model[0])

    # Choose best model and evaluate values on validation data
    print('Evaluating on best model for fold %d'% fold)
    best_model_index = np.argmax(val_acc)
    best_model, best_params, best_model_type = models[best_model_index]
    print('Best model type and parameters:')
    print(best_model_type)
    print(best_params)
  
    nr_epochs = 1
    ntrain = out_X_train.shape[0]; nval = ntrain//5
    val_idx = np.random.randint(ntrain, size=nval)
    train_idx = [i for i in range(out_X_train.shape[0]) if i not in val_idx]
    trainX = out_X_train[train_idx]; trainY = out_y_train[train_idx]
    valX = out_X_train[val_idx]; valY = out_y_train[val_idx]
    history = best_model.fit(trainX, trainY, epochs=nr_epochs, \
                             validation_data=(valX, valY))
    
    # Save model
    best_model.save(os.path.join(resultdir,'best_model_fold'+str(fold)+'.h5'))

    # Predict probability on validation data
    probs = best_model.predict_proba(out_X_test, batch_size=1)
    y_pred = probs.argmax(axis=1)
    y_true = out_y_test.argmax(axis=1)
    conf_mat = pd.crosstab(pd.Series(y_true), pd.Series(y_pred)) 
    conf_mat.index = [sleep_states[idx] for idx in conf_mat.index]
    conf_mat.columns = [sleep_states[idx] for idx in conf_mat.columns]
    conf_mat.reindex(columns=[lbl for lbl in sleep_states], fill_value=0)
    print(conf_mat)

if __name__ == "__main__":
  main(sys.argv[1:])
