import sys,os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from imblearn.over_sampling import SMOTE

def get_features(featfile, mode):
  df = pd.read_csv(featfile, dtype={'label':object, 'user':object,
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
  
  feat_cols = ['ENMO_mean','ENMO_std','ENMO_range','ENMO_mad',
               'ENMO_entropy1','ENMO_entropy2', 'ENMO_prev30diff', 'ENMO_next30diff',
               'ENMO_prev60diff', 'ENMO_next60diff', 'ENMO_prev120diff', 'ENMO_next120diff',
               'angz_mean','angz_std','angz_range','angz_mad',
               'angz_entropy1','angz_entropy2', 'angz_prev30diff', 'angz_next30diff',
               'angz_prev60diff', 'angz_next60diff', 'angz_prev120diff', 'angz_next120diff',
               'LIDS_mean','LIDS_std','LIDS_range','LIDS_mad',
               'LIDS_entropy1','LIDS_entropy2', 'LIDS_prev30diff', 'LIDS_next30diff',
               'LIDS_prev60diff', 'LIDS_next60diff', 'LIDS_prev120diff', 'LIDS_next120diff']

  X = df[feat_cols].values
  y = df['label']
  y = np.array([states.index(i) for i in y])
  return X, y

def main(argv):
  train_cvfile = argv[0]  
  modeldir = argv[1]
  train_featfile = argv[2]
  mode = argv[3]
  outdir = argv[4]

  if not os.path.exists(os.path.join(outdir, 'models')):
    os.makedirs(os.path.join(outdir, 'models'))

  # Read CV results and find best model
  cv_df = pd.read_csv(train_cvfile)
  
  sleep_states = [col.split('_')[1] for col in cv_df.columns if col.startswith('true')]
  sleep_labels = [idx for idx,state in enumerate(sleep_states)]
  true_cols = ['true_'+state for state in sleep_states]
  pred_cols = ['smooth_'+state for state in sleep_states]
  nclasses = len(true_cols)
  nfolds = len(set(cv_df['Fold']))
  best_ap = -1; best_fold = -1
  for fold in range(nfolds):  
    # Get overall scores excluding nonwear
    true_prob = cv_df[cv_df['Fold'] == fold+1][true_cols].values
    valid_indices = true_prob.sum(axis=1) > 0
    true_prob = true_prob[valid_indices]
    y_true = true_prob.argmax(axis=1)
    pred_prob = cv_df[cv_df['Fold'] == fold+1][pred_cols].values[valid_indices]
    pred_prob = pred_prob/pred_prob.sum(axis=1).reshape(-1,1)
    pred_prob[np.isnan(pred_prob)] = 1.0/nclasses
    y_pred = pred_prob.argmax(axis=1)
    prec, rec, fsc, sup = precision_recall_fscore_support(y_true, y_pred, average='macro')
    ap = average_precision_score(true_prob, pred_prob, average='macro')
    print(fold+1, fsc, ap)
    if ap > best_ap:
      best_ap = ap
      best_fold = fold+1
  print('Best fold = Fold %d with AP = %0.4f' % (best_fold, best_ap))

  # Load best model
  model_fnames = os.listdir(modeldir)
  best_model_fname = [fname for fname in model_fnames if ('fold'+str(best_fold) in fname) and mode in fname][0]
  scaler, cv_clf = joblib.load(open(os.path.join(modeldir, best_model_fname), 'rb'))
  clf = cv_clf.best_estimator_
  if mode == 'multiclass':
    class_wt = {}
    for i in range(len(clf.class_weight)):
        class_wt[i] = clf.class_weight[i][1]
    clf.class_weight = class_wt
  if clf.max_depth is not None:
    print('Best model is RF with #trees = %d, depth = %d' % (clf.n_estimators, clf.max_depth))
  else:
    print('Best model is RF with #trees = %d with infinite depth' % (clf.n_estimators))

  # Train the best model with entire train data
  # Before that, standardize training data and balance with SMOTE
  train_X, train_y = get_features(train_featfile, mode)
  train_X_sc = scaler.transform(train_X)
  smote = SMOTE(random_state=0, n_jobs=-1, sampling_strategy='all')
  train_X_resamp, train_y_resamp = smote.fit_resample(train_X_sc, train_y)
  clf.fit(train_X_resamp, train_y_resamp)
 
  # Save trained model
  joblib.dump([scaler, clf], os.path.join(outdir, 'models',\
              'best_'+ mode + '_balanced_RF.sav'))

if __name__ == "__main__":
  main(sys.argv[1:])
