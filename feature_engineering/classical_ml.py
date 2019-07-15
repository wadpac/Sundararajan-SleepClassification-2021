# -*- coding: utf-8 -*-

import sys,os
import time
import pandas as pd
import numpy as np
import scipy
from collections import Counter
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn import manifold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def save_user_report(pred_list, sleep_states, fname):
  nfolds = len(pred_list)
  for i in range(nfolds):
    users = pred_list[i][0]
    y_true = pred_list[i][1]
    y_true = [sleep_states[idx] for idx in y_true]
    y_pred = pred_list[i][2]
    y_pred = [sleep_states[idx] for idx in y_pred]
    fold = np.array([i+1]*users.shape[0])
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
  print('ConfMat\tWake\tNREM1\tNREM2\tNREM3\tREM\n')
  for i in range(confusion_mat.shape[0]):
    #print('%s\t%0.4f' % (sleep_states[i], confusion_mat[i,0]))
    print('%s\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f' % (sleep_states[i], confusion_mat[i][0], confusion_mat[i][1], confusion_mat[i][2], confusion_mat[i][3], confusion_mat[i][4]))
  print('\n')    

def get_feat_importances_report(importances, feature_names):
  indices = np.argsort(importances)[::-1]
  print('Feature ranking:')
  for i in range(importances.shape[0]):
    print('%d. %s: %0.4f' % (i+1,feature_names[indices[i]],importances[indices[i]]))

def main(argv):
  infile = argv[0]
  outdir = argv[1]

  if not os.path.exists(outdir):
    os.makedirs(outdir)

  # Read data file and retain data only corresponding to 5 sleep states
  df = pd.read_csv(infile, dtype={'label':object, 'user':object, 'position':object, 'dataset':object})
  orig_cols = df.columns
  sleep_states = ['Wake','NREM 1','NREM 2','NREM 3','REM']
  df = df[df['label'].isin(sleep_states)].reset_index()
  df = df[df['dataset'] == 'UPenn'].reset_index()
  df = df[orig_cols]
  print('... Number of data samples: %d' % len(df))
  ctr = Counter(df['label'])
  for cls in ctr:
    print('%s: %d (%0.2f%%)' % (cls,ctr[cls],ctr[cls]*100.0/len(df))) 

  feat_cols = ['ENMO_mean','ENMO_std','ENMO_min','ENMO_max','ENMO_mad','ENMO_entropy1','ENMO_entropy2', 'ENMO_prevdiff', 'ENMO_nextdiff', \
               'angz_mean','angz_std','angz_min','angz_max','angz_mad','angz_entropy1','angz_entropy2', 'angz_prevdiff', 'angz_nextdiff', \
               'LIDS_mean','LIDS_std','LIDS_min','LIDS_max','LIDS_mad','LIDS_entropy1','LIDS_entropy2', 'LIDS_prevdiff', 'LIDS_nextdiff']


  ######################## Partition the datasets #######################

  # Nested cross-validation - outer CV for estimating model performance
  # Inner CV for estimating model hyperparameters

  # Split data based on users, not on samples, for outer CV
  # Use Stratified CV for inner CV to ensure similar label distribution
  X = df[feat_cols].values
  y = df['label']
  y = np.array([sleep_states.index(i) for i in y])
  groups = df['user']

  feat_len = X.shape[1]

  # Outer CV
  imbalanced_pred = []; imbalanced_imp = np.zeros(feat_len)
  balanced_pred = []; balanced_imp = np.zeros(feat_len)
  outer_cv_splits = 5; inner_cv_splits = 3
  group_kfold = GroupKFold(n_splits=outer_cv_splits)
  out_fold = 0
  for train_indices, test_indices in group_kfold.split(X,y,groups):
    out_fold += 1
    out_fold_X_train = X[train_indices,:]; out_fold_X_test = X[test_indices,:]
    out_fold_y_train = y[train_indices]; out_fold_y_test = y[test_indices]
    out_fold_users_test = groups[test_indices]

    # Inner CV
    strat_kfold = StratifiedKFold(n_splits=inner_cv_splits, random_state=0, shuffle=True)       
    #################### Without balancing #######################

    custom_cv_indices = []
    for grp_train_idx, grp_test_idx in strat_kfold.split(out_fold_X_train,out_fold_y_train):
      custom_cv_indices.append((grp_train_idx, grp_test_idx))

    pipe = Pipeline([('scl', StandardScaler()), \
                 ('clf', RandomForestClassifier(class_weight='balanced', \
                 random_state=0))])

    print('Fold'+str(out_fold)+' - Imbalanced: Hyperparameter search')
    search_params = {'clf__n_estimators':[50,100,200,300,500], \
                 'clf__max_depth': [5,10,None]}
    cv_clf = RandomizedSearchCV(estimator=pipe, param_distributions=search_params, \
                            cv=custom_cv_indices, scoring='f1_macro', n_iter=5, \
                            n_jobs=-1, verbose=2)
    cv_clf.fit(out_fold_X_train, out_fold_y_train)
    out_fold_y_test_pred = cv_clf.predict(out_fold_X_test)
    print('Fold'+str(out_fold)+' - Imbalanced', cv_clf.best_params_)

    imbalanced_pred.append((out_fold_users_test, out_fold_y_test, out_fold_y_test_pred))
    imbalanced_imp = imbalanced_imp + cv_clf.best_estimator_.named_steps['clf'].feature_importances_

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
    #enn = EditedNearestNeighbours(random_state=0, n_jobs=-1, sampling_strategy='all')
    #smote_enn = SMOTEENN(smote=smote, enn=enn,
    #                     random_state=0, sampling_strategy='all')
    out_fold_X_train_resamp, out_fold_y_train_resamp = smote.fit_resample(out_fold_X_train_sc, out_fold_y_train)

    custom_resamp_cv_indices = []
    for grp_train_idx, grp_test_idx in strat_kfold.split(out_fold_X_train_resamp,out_fold_y_train_resamp):
      custom_resamp_cv_indices.append((grp_train_idx, grp_test_idx))

    # Note: imblearn Pipeline is slow and sklearn pipeline yields poor results 
    clf = RandomForestClassifier(class_weight='balanced', \
                             max_depth=None, random_state=0)

    print('Fold'+str(out_fold)+' - Balanced: Hyperparameter search')
    search_params = {'n_estimators':[50,100,200,300,500], \
                 'max_depth': [5,10,None]}
    cv_clf = RandomizedSearchCV(estimator=clf, param_distributions=search_params, \
                            cv=custom_resamp_cv_indices, scoring='f1_macro', \
                            n_iter=5, n_jobs=-1, verbose=2)
    cv_clf.fit(out_fold_X_train_resamp, out_fold_y_train_resamp)
    out_fold_y_test_pred = cv_clf.predict(out_fold_X_test_sc)
    print('Fold'+str(out_fold)+' - Balanced', cv_clf.best_params_)

    balanced_pred.append((out_fold_users_test, out_fold_y_test, out_fold_y_test_pred))
    balanced_imp = balanced_imp + cv_clf.best_estimator_.feature_importances_

  # Get imbalanced classification reports
  print('############## Imbalanced classification ##############')
  get_classification_report(imbalanced_pred, sleep_states)
  imbalanced_imp = imbalanced_imp / float(outer_cv_splits)
  get_feat_importances_report(imbalanced_imp, feat_cols)

  # Save imbalanced prediction results for every user
  save_user_report(imbalanced_pred, sleep_states, os.path.join(outdir,'imbalanced_results.csv')) 

  # Get balanced classification reports
  print('############## Balanced classification ##############')
  get_classification_report(balanced_pred, sleep_states)
  balanced_imp = balanced_imp / float(outer_cv_splits)
  get_feat_importances_report(balanced_imp, feat_cols)

  # Save balanced prediction results for every user
  save_user_report(balanced_pred, sleep_states, os.path.join(outdir,'balanced_results.csv')) 
    
if __name__ == "__main__":
    main(sys.argv[1:])
