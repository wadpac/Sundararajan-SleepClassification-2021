# -*- coding: utf-8 -*-

import sys,os
import pandas as pd
import numpy as np
from collections import Counter
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

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
  imbalanced_pred = []; imbalanced_imp = []
  balanced_pred = []; balanced_imp = []
  outer_cv_splits = 5; inner_cv_splits = 5
  group_kfold = GroupKFold(n_splits=outer_cv_splits)
  out_fold = 0
  for train_indices, test_indices in group_kfold.split(X,y,groups):
    out_fold += 1
    out_fold_X_train = X[train_indices,:]; out_fold_X_test = X[test_indices,:]
    out_fold_y_train = y[train_indices]; out_fold_y_test = y[test_indices]
    out_fold_users_test = groups[test_indices]
    out_fold_ts_test = ts[test_indices]
    out_fold_fnames_test = fnames[test_indices]

    # Inner CV
    strat_kfold = StratifiedKFold(n_splits=inner_cv_splits, random_state=0,
                                  shuffle=True)       
#    #################### Without balancing #######################
#
#    custom_cv_indices = []
#    for grp_train_idx, grp_test_idx in \
#            strat_kfold.split(out_fold_X_train,out_fold_y_train):
#      custom_cv_indices.append((grp_train_idx, grp_test_idx))
#
#    pipe = Pipeline([('scl', StandardScaler()),
#                 ('clf', RandomForestClassifier(class_weight='balanced',
#                 random_state=0))])
#
#    print('Fold'+str(out_fold)+' - Imbalanced: Hyperparameter search')
#    search_params = {'clf__n_estimators':[50,100,200,300,500],
#                 'clf__max_depth': [5,10,None]}
#    cv_clf = RandomizedSearchCV(estimator=pipe, param_distributions=search_params,
#                            cv=custom_cv_indices, scoring='f1_macro', n_iter=5,
#                            n_jobs=-1, verbose=2)
#    cv_clf.fit(out_fold_X_train, out_fold_y_train)
#    pickle.dump(cv_clf, open(os.path.join(resultdir,\
#                'fold'+str(out_fold)+'_'+ mode + '_imbalanced_RF.sav'),'wb'))
#    out_fold_y_test_pred = cv_clf.predict_proba(out_fold_X_test)
#    print('Fold'+str(out_fold)+' - Imbalanced', cv_clf.best_params_)
#
#    imbalanced_pred.append((out_fold_users_test, out_fold_ts_test, out_fold_fnames_test,
#                            out_fold_y_test, out_fold_y_test_pred))
#    imbalanced_imp.append(cv_clf.best_estimator_.named_steps['clf'].feature_importances_)

    ################## Balancing with SMOTE ###################

    scaler = StandardScaler()
    scaler.fit(out_fold_X_train)
    out_fold_X_train_sc = scaler.transform(out_fold_X_train)
    out_fold_X_test_sc = scaler.transform(out_fold_X_test)

    # Resample training data
    print('Fold'+str(out_fold)+' - Balanced: SMOTE')
    # Imblearn - Undersampling techniques ENN and Tomek are too slow and 
    # difficult to parallelize
    # So stick only with oversampling techniques
    smote = SMOTE(random_state=0, n_jobs=-1, sampling_strategy='all')
    out_fold_X_train_resamp, out_fold_y_train_resamp = \
                    smote.fit_resample(out_fold_X_train_sc, out_fold_y_train)

    custom_resamp_cv_indices = []
    for grp_train_idx, grp_test_idx in \
          strat_kfold.split(out_fold_X_train_resamp,out_fold_y_train_resamp):
      custom_resamp_cv_indices.append((grp_train_idx, grp_test_idx))

    # Note: imblearn Pipeline is slow and sklearn pipeline yields poor results 
    clf = RandomForestClassifier(class_weight='balanced',
                             max_depth=None, random_state=0)

    print('Fold'+str(out_fold)+' - Balanced: Hyperparameter search')
    search_params = {'n_estimators':[50,100,200,300,500],
                 'max_depth': [5,10,None]}
    cv_clf = RandomizedSearchCV(estimator=clf, param_distributions=search_params,
                            cv=custom_resamp_cv_indices, scoring='f1_macro',
                            n_iter=5, n_jobs=-1, verbose=2)
    cv_clf.fit(out_fold_X_train_resamp, out_fold_y_train_resamp)
    pickle.dump([scaler,cv_clf], open(os.path.join(resultdir,\
                'fold'+str(out_fold)+'_'+ mode + '_balanced_RF.sav'),'wb'))
    out_fold_y_test_pred = cv_clf.predict_proba(out_fold_X_test_sc)
    print('Fold'+str(out_fold)+' - Balanced', cv_clf.best_params_)

    balanced_pred.append((out_fold_users_test, out_fold_ts_test, out_fold_fnames_test,
                          out_fold_y_test, out_fold_y_test_pred))
    balanced_imp.append(cv_clf.best_estimator_.feature_importances_)

#  print('############## Imbalanced classification ##############')
#  # Save imbalanced classification reports
#  cv_save_feat_importances_result(imbalanced_imp, feat_cols,
#                   os.path.join(outdir, mode + '_imbalanced_feat_imp.csv'))
#  cv_save_classification_result(imbalanced_pred, states,
#                   os.path.join(outdir, mode + '_imbalanced_classification.csv'))
 
  print('############## Balanced classification ##############')
  # Save balanced classification reports
  cv_save_feat_importances_result(balanced_imp, feat_cols,
                   os.path.join(outdir, mode + '_balanced_feat_imp.csv'))
  cv_save_classification_result(balanced_pred, states,
                   os.path.join(outdir, mode + '_balanced_classification.csv'))
    
if __name__ == "__main__":
    main(sys.argv[1:])
