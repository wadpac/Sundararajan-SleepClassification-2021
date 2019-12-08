import sys,os
import pandas as pd
import argparse
import sklearn
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

sys.path.append('../analysis/')
from analysis import cv_save_feat_importances_result, cv_save_classification_result

def get_data(fname, feat_cols, sleep_states, mode='binary'):
  df = pd.read_csv(fname)  
  if mode == 'binary':
    df.loc[df['label'].isin(['REM','NREM 1','NREM 2','NREM 3']), 'label'] = 'Sleep'
  df = df[df['label'].isin(sleep_states)].reset_index()
  X = df[feat_cols].values
  y = df['label']
  y = np.array([sleep_states.index(val) for val in y])
  users = list(df['user'])
  timestamp = list(df['timestamp'])
  fnames = list(df['filename'])
  return X, y, users, timestamp, fnames

def main(args):
  if not os.path.exists(os.path.join(args.outdir, 'models')):
    os.makedirs(os.path.join(args.outdir, 'models'))

  if args.mode == 'binary':
    sleep_states = ['Wake','Sleep','Nonwear']
  else:
    sleep_states = ['Wake','NREM 1','NREM 2','NREM 3','REM','Nonwear']

  feat_cols = ['ENMO_mean','ENMO_std','ENMO_min','ENMO_max','ENMO_mad',
               'ENMO_entropy1','ENMO_entropy2', 'ENMO_prevdiff', 'ENMO_nextdiff',
               'angz_mean','angz_std','angz_min','angz_max','angz_mad',
               'angz_entropy1','angz_entropy2', 'angz_prevdiff', 'angz_nextdiff',
               'LIDS_mean','LIDS_std','LIDS_min','LIDS_max','LIDS_mad',
               'LIDS_entropy1','LIDS_entropy2', 'LIDS_prevdiff', 'LIDS_nextdiff']

  X_test, y_test, users_test, ts_test, fnames_test = get_data(args.test, feat_cols, sleep_states, mode=args.mode)
  predictions = []
  feat_imp = []
  if args.testmode == 'pretrain': 
    # 'pretrained' - use pretrained models on test data
    model_files = os.listdir(args.modeldir)
    for i,fname in enumerate(model_files):
      scaler, clf = pickle.load(open(os.path.join(args.modeldir,fname), 'rb'))
      X_test_sc = scaler.transform(X_test)
      y_pred = clf.predict_proba(X_test_sc)
      predictions.append((users_test, ts_test, fnames_test, y_test, y_pred))
      feat_imp.append(clf.best_estimator_.feature_importances_)
  else:
    # 'finetune'- use models tuned using validation data from same distribution as test data
    X_train, y_train, _, _, _ = get_data(args.train, feat_cols, sleep_states, mode=args.mode)
    X_val, y_val, _, _, _ = get_data(args.val, feat_cols, sleep_states, mode=args.mode)

    # Scale features
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    # Balance training samples using SMOTE
    smote = SMOTE(random_state=0, n_jobs=-1, sampling_strategy='all')
    X_train_resamp, y_train_resamp = smote.fit_resample(X_train_sc, y_train)
    X_concat = np.concatenate((X_train_resamp, X_val_sc), axis=0)
    y_concat = np.concatenate((y_train_resamp, y_val), axis=0)

    # Get suitable parameters using validation data
    clf = RandomForestClassifier(class_weight='balanced',
                             max_depth=None, random_state=0)
    search_params = {'n_estimators':[50,100,200,300,500],
                 'max_depth': [5,10,None]}
    cv_indices = [(range(X_train_sc.shape[0]), range(X_train_sc.shape[0], X_concat.shape[0]))]
    cv_clf = RandomizedSearchCV(estimator=clf, param_distributions=search_params,
                            cv=cv_indices, scoring='f1_macro',
                            n_iter=10, n_jobs=-1, verbose=2)
    cv_clf.fit(X_concat, y_concat)
    pickle.dump(cv_clf, open(os.path.join(args.outdir, 'models',
                'transfer_' + args.mode +'_'+ args.testmode + '_RF.sav'),'wb'))
    y_pred = cv_clf.predict_proba(X_test_sc)

    predictions.append((users_test, ts_test, fnames_test, y_test, y_pred))
    feat_imp.append(cv_clf.best_estimator_.feature_importances_)

  cv_save_feat_importances_result(feat_imp, feat_cols,
                   os.path.join(args.outdir, 'transfer_' + args.mode + '_' + args.testmode + '_feat_imp.csv'))
  cv_save_classification_result(predictions, sleep_states,
                   os.path.join(args.outdir, 'transfer_' + args.mode + '_' + args.testmode + '_classification.csv'))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--testmode', type=str, default='pretrain',
                      help='mode of transfer learning - pretrain or finetune')
  parser.add_argument('--modeldir', type=str, help='input directory to load pre-trained models')
  parser.add_argument('--mode', type=str, default='binary', help='classification mode - binary/multiclass')
  parser.add_argument('--train', type=str, help='training data file for finetune')        
  parser.add_argument('--val', type=str, help='validation data file for finetune')        
  parser.add_argument('--test', type=str, help='test data file for pretrain/finetune')        
  parser.add_argument('--outdir', type=str, help='output directory to save results')
  args = parser.parse_args()
  
  main(args)
