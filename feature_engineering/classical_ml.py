# -*- coding: utf-8 -*-

import sys,os
import pandas as pd
import numpy as np
from collections import Counter
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import make_scorer, average_precision_score

from imblearn.over_sampling import SMOTE

sys.path.append('../analysis/')
from analysis import cv_save_feat_importances_result, cv_save_classification_result

def main(argv):
  infile = argv[0]
  mode = argv[1] # binary or multiclass or nonwear
  dataset = argv[2]
  outdir = argv[3]

  resultdir = os.path.join(outdir,'models')
  if not os.path.exists(resultdir):
    os.makedirs(resultdir)

  # Read data file and retain data only corresponding to 5 sleep states or nonwear
  df = pd.read_csv(infile, dtype={'label':object, 'user':object,
                   'position':object, 'dataset':object})
  if mode == 'binary':
    states = ['Wake', 'Sleep']
    collate_states = ['NREM 1', 'NREM 2', 'NREM 3', 'REM']
    df.loc[df['label'].isin(collate_states), 'label'] = 'Sleep'
  elif mode == 'nonwear':
    states = ['Wear', 'Nonwear']
    collate_states = ['Wake', 'NREM 1', 'NREM 2', 'NREM 3', 'REM']
    df.loc[df['label'].isin(collate_states), 'label'] = 'Wear'
  else:
    states = ['Wake', 'NREM 1', 'NREM 2', 'NREM 3', 'REM']
    
  df = df[df['label'].isin(states)].reset_index()
  
  print('... Number of data samples: %d' % len(df))
  ctr = Counter(df['label'])
  for cls in ctr:
    print('%s: %d (%0.2f%%)' % (cls,ctr[cls],ctr[cls]*100.0/len(df))) 

  feat_cols = ['ENMO_mean','ENMO_std','ENMO_range','ENMO_mad',
               'ENMO_entropy1','ENMO_entropy2', 'ENMO_prev30diff', 'ENMO_next30diff',
               'ENMO_prev60diff', 'ENMO_next60diff', 'ENMO_prev120diff', 'ENMO_next120diff',
               'angz_mean','angz_std','angz_range','angz_mad',
               'angz_entropy1','angz_entropy2', 'angz_prev30diff', 'angz_next30diff',
               'angz_prev60diff', 'angz_next60diff', 'angz_prev120diff', 'angz_next120diff',
               'LIDS_mean','LIDS_std','LIDS_range','LIDS_mad',
               'LIDS_entropy1','LIDS_entropy2', 'LIDS_prev30diff', 'LIDS_next30diff',
               'LIDS_prev60diff', 'LIDS_next60diff', 'LIDS_prev120diff', 'LIDS_next120diff']

  ######################## Partition the datasets #######################

  # Nested cross-validation - outer CV for estimating model performance
  # Inner CV for estimating model hyperparameters

  # Split data based on users, not on samples, for outer CV
  # Use Stratified CV for inner CV to ensure similar label distribution
  ts = df['timestamp']
  X = df[feat_cols].values
  y = df['label']
  y = np.array([states.index(i) for i in y])
  groups = df['user']
  fnames = df['filename']

  feat_len = X.shape[1]

  # Outer CV
  scorer = make_scorer(average_precision_score, average='macro')
  imbalanced_pred = []; imbalanced_imp = []
  balanced_pred = []; balanced_imp = []
  outer_cv_splits = 5; inner_cv_splits = 5
  outer_group_kfold = GroupKFold(n_splits=outer_cv_splits)
  out_fold = 0
  for train_indices, test_indices in outer_group_kfold.split(X,y,groups):
    out_fold += 1
    out_fold_X_train = X[train_indices,:]; out_fold_X_test = X[test_indices,:]
    out_fold_y_train = y[train_indices]; out_fold_y_test = y[test_indices]
    out_fold_users_train = groups[train_indices]; out_fold_users_test = groups[test_indices]
    out_fold_ts_test = ts[test_indices]
    out_fold_fnames_test = fnames[test_indices]

    class_wt = compute_class_weight('balanced', np.unique(out_fold_y_train), out_fold_y_train)
    class_wt = {i:val for i,val in enumerate(class_wt)}

    # Inner CV
    ################## Balancing with SMOTE ###################
    scaler = StandardScaler()
    scaler.fit(out_fold_X_train)
    out_fold_X_train_sc = scaler.transform(out_fold_X_train)
    out_fold_X_test_sc = scaler.transform(out_fold_X_test)
    
    # Imblearn - Undersampling techniques ENN and Tomek are too slow and 
    # difficult to parallelize
    # So stick only with oversampling techniques
    print('Fold'+str(out_fold)+' - Balanced: SMOTE')
    smote = SMOTE(random_state=0, n_jobs=-1, sampling_strategy='all')
    # Resample training data for each user
    train_users = list(set(out_fold_users_train))
    out_fold_X_train_resamp, out_fold_y_train_resamp, out_fold_users_train_resamp = None, None, None
    for i,user in enumerate(train_users):
      #print('%d/%d - %s' % (i+1,len(train_users),user))
      user_X = out_fold_X_train_sc[out_fold_users_train == user]
      user_y = out_fold_y_train[out_fold_users_train == user]
      if len(set(user_y)) == 1:
        print('%d/%d: %s has only one class' % (i+1,len(train_users),user))
        print(Counter(user_y))
        continue
      try:
        user_X_resamp, user_y_resamp = smote.fit_resample(user_X, user_y)
      except:
        print('%d/%d: %s failed to fit' % (i+1,len(train_users),user))
        print(Counter(user_y))
        continue
      user_y_resamp = user_y_resamp.reshape(-1,1)
      user_resamp = np.array([user] * len(user_X_resamp)).reshape(-1,1)
      if out_fold_X_train_resamp is None:
        out_fold_X_train_resamp = user_X_resamp
        out_fold_y_train_resamp = user_y_resamp
        out_fold_users_train_resamp = user_resamp
      else:
        out_fold_X_train_resamp = np.vstack((out_fold_X_train_resamp, user_X_resamp))
        out_fold_y_train_resamp = np.vstack((out_fold_y_train_resamp, user_y_resamp))
        out_fold_users_train_resamp = np.vstack((out_fold_users_train_resamp, user_resamp))
    # Shuffle resampled data
    resamp_indices = np.arange(len(out_fold_X_train_resamp))
    np.random.shuffle(resamp_indices)
    out_fold_X_train_resamp = out_fold_X_train_resamp[resamp_indices]
    out_fold_y_train_resamp = out_fold_y_train_resamp[resamp_indices].reshape(-1)
    out_fold_users_train_resamp = out_fold_users_train_resamp[resamp_indices].reshape(-1)

    inner_group_kfold = GroupKFold(n_splits=inner_cv_splits)
    custom_resamp_cv_indices = []
    for grp_train_idx, grp_test_idx in \
          inner_group_kfold.split(out_fold_X_train_resamp, out_fold_y_train_resamp, out_fold_users_train_resamp):
      custom_resamp_cv_indices.append((grp_train_idx, grp_test_idx))
      grp_train_users = out_fold_users_train_resamp[grp_train_idx]
      grp_test_users = out_fold_users_train_resamp[grp_test_idx]

    # Note: imblearn Pipeline is slow and sklearn pipeline yields poor results 
    clf = RandomForestClassifier(class_weight=class_wt,
                             max_depth=None, random_state=0)

    print('Fold'+str(out_fold)+' - Balanced: Hyperparameter search')
    search_params = {'n_estimators':[100,150,200,300,400,500],
                 'max_depth': [5,10,15,20,None]}
    cv_clf = RandomizedSearchCV(estimator=clf, param_distributions=search_params,
                            cv=custom_resamp_cv_indices, scoring=scorer,
                            n_iter=10, n_jobs=-1, verbose=2)
    cv_clf.fit(out_fold_X_train_resamp, out_fold_y_train_resamp)
    print(cv_clf.best_estimator_)
    joblib.dump([scaler,cv_clf], os.path.join(resultdir,\
                'fold'+str(out_fold)+'_'+ mode + '_balanced_RF.sav'))
    out_fold_y_test_pred = cv_clf.predict_proba(out_fold_X_test_sc)
    print('Fold'+str(out_fold)+' - Balanced', cv_clf.best_params_)

    balanced_pred.append((out_fold_users_test, out_fold_ts_test, out_fold_fnames_test,
                          out_fold_y_test, out_fold_y_test_pred))
    balanced_imp.append(cv_clf.best_estimator_.feature_importances_)

  print('############## Balanced classification ##############')
  # Save balanced classification reports
  cv_save_feat_importances_result(balanced_imp, feat_cols,
                   os.path.join(outdir, mode + '_balanced_feat_imp.csv'))
  cv_save_classification_result(balanced_pred, states,
                   os.path.join(outdir, mode + '_balanced_classification.csv'))
    
if __name__ == "__main__":
    main(sys.argv[1:])
