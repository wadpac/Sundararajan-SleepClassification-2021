import sys,os
import joblib
import numpy as np
import pandas as pd

import networkx as nx
from networkx import DiGraph

sys.path.append('../../sklearn-hierarchical-classification/')
from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.metrics import multi_labeled, fill_ancestors

sys.path.append('../analysis/')
from analysis import cv_save_classification_result

def main(argv):
  infile = argv[0]
  modeldir = argv[1]
  mode = argv[2]
  ensemble = int(argv[3]) # 0 - use best model, 1 - use ensemble
  outdir = argv[4]

  df = pd.read_csv(infile)
  method = 'feat_eng'
  if mode == 'binary':
    states = ['Wake', 'Sleep']
    collate_states = ['NREM 1', 'NREM 2', 'NREM 3', 'REM']
    df.loc[df['label'].isin(collate_states), 'label'] = 'Sleep'
  elif mode == 'nonwear':
    states = ['Wear', 'Nonwear']
    collate_states = ['Wake', 'NREM 1', 'NREM 2', 'NREM 3', 'REM']
    df.loc[df['label'].isin(collate_states), 'label'] = 'Wear'
  elif mode == 'multiclass':
    states = ['Wake', 'NREM 1', 'NREM 2', 'NREM 3', 'REM']
  elif mode == 'hierarchical':
    method = 'hierarchical'
    states = ['Wake', 'NREM 1', 'NREM 2', 'NREM 3', 'REM','Nonwear']
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

  df = df[df['label'].isin(states)].reset_index()
  
  feat_cols = ['ENMO_mean','ENMO_std','ENMO_range','ENMO_mad',
               'ENMO_entropy1','ENMO_entropy2', 'ENMO_prev30diff', 'ENMO_next30diff',
               'ENMO_prev60diff', 'ENMO_next60diff', 'ENMO_prev120diff', 'ENMO_next120diff',
               'angz_mean','angz_std','angz_range','angz_mad',
               'angz_entropy1','angz_entropy2', 'angz_prev30diff', 'angz_next30diff',
               'angz_prev60diff', 'angz_next60diff', 'angz_prev120diff', 'angz_next120diff',
               'LIDS_mean','LIDS_std','LIDS_range','LIDS_mad',
               'LIDS_entropy1','LIDS_entropy2', 'LIDS_prev30diff', 'LIDS_next30diff',
               'LIDS_prev60diff', 'LIDS_next60diff', 'LIDS_prev120diff', 'LIDS_next120diff']

  ts_test = df['timestamp']
  x_test = df[feat_cols].values
  y_test = df['label']
  if mode != 'hierarchical':
    y_test = np.array([states.index(i) for i in y_test])
  users_test = df['user']
  fnames_test = df['filename']

  N = x_test.shape[0]

  if ensemble:
    model_fnames = os.listdir(modeldir)
    model_fnames = [fname for fname in model_fnames if mode in fname]
    nfolds = len(model_fnames)
    for fold,fname in enumerate(model_fnames):
      print('Processing fold ' + str(fold+1))
      if mode != 'hierarchical':
        scaler, cv_clf = joblib.load(open(os.path.join(modeldir, fname), 'rb'))
        x_test_sc = scaler.transform(x_test)
        fold_y_pred = cv_clf.predict_proba(x_test_sc)
      else:
        cv_clf = pickle.load(open(os.path.join(modeldir, fname), 'rb'))
        cv_clf = cv_clf.best_estimator_
        fold_y_pred = cv_clf.predict(x_test)
        fold_y_pred_prob = cv_clf.predict_proba(x_test)
        with multi_labeled(y_test, fold_y_pred, cv_clf.named_steps['clf'].graph_) \
                              as (y_test_, y_pred_, graph_, classes_):
          states = classes_ 
          y_test_ = fill_ancestors(y_test_, graph=graph_)
          fold_y_pred_ = np.zeros(fold_y_pred_prob.shape)
          for new_idx, label in enumerate(classes_):
            old_idx = classes.index(label)
            fold_y_pred_[:,new_idx] = fold_y_pred_prob[:,old_idx]
        fold_y_pred = fold_y_pred_
  
      # Accumulate prediction probabilities
      if fold == 0:
        y_pred = np.zeros((N,len(states)))
      y_pred += fold_y_pred
    
    # Get average predictions
    y_pred = y_pred/float(nfolds)
    if mode == 'hierarchical':
      y_test = y_test_
  else:
    if mode != 'hierarchical':  
      model_fnames = os.listdir(modeldir)
      model_fname = [fname for fname in model_fnames if mode in fname][0]  
      scaler, clf = joblib.load(open(os.path.join(modeldir, model_fname), 'rb'))
      x_test_sc = scaler.transform(x_test)
      y_pred = clf.predict_proba(x_test_sc)

  # Save test results
  y_pred = [(users_test, ts_test, fnames_test, y_test, y_pred)]
  cv_save_classification_result(y_pred, states,
                   os.path.join(outdir, mode + '_test_classification.csv'),
                   method = method)
  
if __name__ == "__main__":
  main(sys.argv[1:])
