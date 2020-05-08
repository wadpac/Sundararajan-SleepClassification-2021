import sys,os
import pandas as pd
import numpy as np
from collections import Counter
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import GroupKFold, StratifiedKFold

sys.path.append('../sklearn-hierarchical-classification/')
from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.metrics import h_fbeta_score, multi_labeled, fill_ancestors

sys.path.append('../analysis/')
from analysis import cv_save_classification_result, custom_h_fbeta
from tqdm import tqdm
import networkx as nx
from networkx import DiGraph
from networkx.drawing.nx_agraph import graphviz_layout

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(argv):
  infile = argv[0]
  dataset = argv[1]
  outdir = argv[2]

  resultdir = os.path.join(outdir, 'models')
  if not os.path.exists(resultdir):
    os.makedirs(resultdir)

  # Read data file and retain data only corresponding to 5 sleep states
  df = pd.read_csv(infile, dtype={'label':object, 'user':object,\
                   'position':object, 'dataset':object})
  states = ['Wake','NREM 1','NREM 2','NREM 3','REM','Nonwear']
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

  ts = df['timestamp']
  X = df[feat_cols].values
  y = df['label']
  #y = np.array([states.index(i) for i in y])
  groups = df['user']
  fnames = df['filename']
  feat_len = X.shape[1]

  # Class hierarchy for sleep stages
  class_hierarchy = {
    ROOT : {"Wear", "Nonwear"},
    "Wear" : {"Wake", "Sleep"},
    "Sleep" : {"NREM", "REM"},
    "NREM" : {"Light", "NREM 3"},
    "Light" : {"NREM 1", "NREM 2"} 
  }
  
  graph = DiGraph(class_hierarchy)    
  classes = [node for node in graph.nodes if node != ROOT]
 
  outer_cv_splits = 5; inner_cv_splits = 5
  factor = 10.0
  
  # Outer CV
  group_kfold = GroupKFold(n_splits=outer_cv_splits)
  out_fold = 0
  hierarchical_pred = []
  for train_indices, test_indices in group_kfold.split(X,y,groups):
    out_fold += 1
    print('Processing fold ' + str(out_fold))
    out_fold_X_train = X[train_indices,:]; out_fold_X_test = X[test_indices,:]
    out_fold_y_train = y[train_indices]; out_fold_y_test = y[test_indices]
    out_fold_users_test = groups[test_indices]
    out_fold_ts_test = ts[test_indices]
    out_fold_fnames_test = fnames[test_indices]
    
    # Create a pipeline with scaler and hierarchical classifier
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('clf', HierarchicalClassifier(
                        base_estimator=RandomForestClassifier(random_state=0, n_estimators=100, n_jobs=-1),
                        class_hierarchy=class_hierarchy,
                        prediction_depth='mlnp',
                        progress_wrapper=tqdm,
                        #stopping_criteria=0.7
                     ))
                    ])
    
    # Inner CV
    strat_kfold = StratifiedKFold(n_splits=inner_cv_splits,\
                                  random_state=0, shuffle=True)       

    custom_cv_indices = []
    for grp_train_idx, grp_test_idx in strat_kfold.split(out_fold_X_train,out_fold_y_train):
      custom_cv_indices.append((grp_train_idx, grp_test_idx))
        
    print('Training')        
    search_params = {'clf__base_estimator__n_estimators':[50,100,200,300,500,700], \
         'clf__base_estimator__max_depth': [5,10,15,None]}
    cv_clf = RandomizedSearchCV(estimator=pipe, param_distributions=search_params, \
                       cv=custom_cv_indices, scoring=make_scorer(custom_h_fbeta,graph=graph), n_iter=5, \
                       n_jobs=-1, verbose=1)
    cv_clf.fit(out_fold_X_train, out_fold_y_train)
    joblib.dump(cv_clf, os.path.join(resultdir,\
                'fold'+str(out_fold)+'_hierarchical_RF.sav'))
    print('Predicting')
    out_fold_y_pred = cv_clf.predict(out_fold_X_test)
    out_fold_y_pred_prob = cv_clf.predict_proba(out_fold_X_test)
    
    best_clf = cv_clf.best_estimator_
        
    # Demonstrate using our hierarchical metrics module with MLB wrapper
    with multi_labeled(out_fold_y_test, out_fold_y_pred, best_clf.named_steps['clf'].graph_) \
                            as (y_test_, y_pred_, graph_, classes_):
      states = classes_ 
      y_test_ = fill_ancestors(y_test_, graph=graph_)
      y_pred_ = fill_ancestors(y_pred_, graph=graph_)
      y_pred_prob_ = np.zeros(out_fold_y_pred_prob.shape)
      for new_idx, label in enumerate(classes_):
        old_idx = classes.index(label)
        y_pred_prob_[:,new_idx] = out_fold_y_pred_prob[:,old_idx]

      hierarchical_pred.append((out_fold_users_test, out_fold_ts_test, out_fold_fnames_test,
                                y_test_, y_pred_prob_))

  cv_save_classification_result(hierarchical_pred, states,
                                os.path.join(outdir, 'hierarchical_classification_results.csv'),
                                method = 'hierarchical')

if __name__ == "__main__":
  main(sys.argv[1:])
