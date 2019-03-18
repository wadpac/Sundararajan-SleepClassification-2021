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
from sklearn import manifold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def get_classification_report(y_true, y_pred, sleep_states):
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    print('Precision = %0.4f' % (precision*100.0))
    print('Recall = %0.4f' % (recall*100.0))
    print('F-score = %0.4f' % (fscore*100.0))
    print('Accuracy = %0.4f' % (accuracy*100.0))
    print(classification_report(y_true, y_pred, target_names=sleep_states))

def get_feat_importances(clf, feature_names):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print('Feature ranking:')
    for i in range(importances.shape[0]):
        print('%d. %s: %0.4f' % (i,feature_names[indices[i]],importances[indices[i]]))

def get_users(infile):
    users = []
    with open(infile,'r') as fp:
      for line in fp:
        users.append(line.strip())
    return users

def main(argv):
    infile = argv[0]
    indir = argv[1]   
    outdir = argv[2]

    if not os.path.exists(outdir):
        os.makedirs(outdir)
 
    # Read data file and retain data only corresponding to 5 sleep states
    df = pd.read_csv(infile, dtype={'label':object, 'user':object, 'position':object, 'dataset':object})
    orig_cols = df.columns
    sleep_states = ['Wake','NREM 1','NREM 2','NREM 3','REM']
    df = df[df['label'].isin(sleep_states)].reset_index()
    df = df[orig_cols]
    print('... Number of data samples: %d' % len(df))
    ctr = Counter(df['label'])
    for cls in ctr:
      print('%s: %d (%0.2f%%)' % (cls,ctr[cls],ctr[cls]*100.0/len(df))) 

    feat_cols = ['ENMO_mean','ENMO_std','ENMO_min','ENMO_max','ENMO_mad','ENMO_entropy1','ENMO_entropy2', \
                 'angz_mean','angz_std','angz_min','angz_max','angz_mad','angz_entropy1','angz_entropy2', \
                 'LIDS_mean','LIDS_std','LIDS_min','LIDS_max','LIDS_mad','LIDS_entropy1','LIDS_entropy2'
]

    ######################## Partition the datasets #######################

    # Split data into train/validation based on users, not on samples
    # Use similar split up for Newcastle and UPenn datasets such that both 
    # are present in all partitions
    print('... Partitioning data')
    users_train = get_users(os.path.join(indir,'users_train.txt'))
    users_val = get_users(os.path.join(indir,'users_val.txt'))

    df_train = df[df['user'].isin(users_train)]
    X_train = df_train[feat_cols].values
    y_train = df_train['label'].values
    y_train = np.array([sleep_states.index(y) for y in y_train])

    df_val = df[df['user'].isin(users_val)]
    X_val = df_val[feat_cols].values
    y_val = df_val['label'].values
    y_val = np.array([sleep_states.index(y) for y in y_val])

    ################## Perform ML without balancing dataset #####################
    
    print('\n... Without balancing data ...')
    start_time = time.time()
    y_train_str = [sleep_states[y] for y in y_train]
    print(Counter(y_train_str))

    pipe = Pipeline([('scl', StandardScaler()), \
                     ('clf', RandomForestClassifier(class_weight='balanced', \
                     n_estimators=100, max_depth=None, \
                     random_state=0))])
    
    # Perform random search for hyper-parameter tuning to find best estimator using validation data
    # Fit training data to best estimator and compute metrics on validation data
    print('... Searching for suitable hyperparameters')
    search_params = {'clf__n_estimators':[50,100,150,200,250,300,500], \
                     'clf__max_depth': [5,10,20,None]}
    cv_clf = RandomizedSearchCV(estimator=pipe, param_distributions=search_params, \
                                cv=3, scoring='f1_macro', n_iter=10, n_jobs=-1)
    cv_clf.fit(X_train, y_train)
    print(cv_clf.best_params_)

    print('... Predicting output with best estimator')
    y_val_pred = cv_clf.predict(X_val)
    get_classification_report(y_val, y_val_pred, sleep_states)
    get_feat_importances(cv_clf.best_estimator_.named_steps['clf'], feat_cols)

    best_model = {'scl_clf':cv_clf.best_estimator_}
    pickle.dump(best_model, open(os.path.join(outdir,'no_balancing.pkl'),'wb'))   

    end_time = time.time()
    print('Time elapsed: %0.2fs' % (end_time-start_time))
    
    ############### Perform ML after balancing dataset with SMOTE ###############
    
    print('\n... After balancing data with SMOTEENN ...')
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_val_sc = scaler.transform(X_val)

    # Resample training data
    start_time = time.time()
    smote_enn = SMOTEENN(random_state=0, sampling_strategy='not majority')
    X_train_resamp_sc, y_train_resamp = smote_enn.fit_resample(X_train_sc, y_train)
    y_train_resamp_str = [sleep_states[y] for y in y_train_resamp]
    print(Counter(y_train_resamp_str))

    # Note: imblearn Pipeline is slow and sklearn pipeline yields poor results 
    clf = RandomForestClassifier(class_weight='balanced', \
                                 n_estimators=100, max_depth=None, \
                                 random_state=0)

    print('... Searching for suitable hyperparameters')
    search_params = {'n_estimators':[50,100,150,200,250,300,500], \
                     'max_depth': [5,10,20,None]}
    cv_clf = RandomizedSearchCV(estimator=clf, param_distributions=search_params, \
                                cv=3, scoring='f1_macro', n_iter=10, n_jobs=-1)
    cv_clf.fit(X_train_resamp_sc, y_train_resamp)
    print(cv_clf.best_params_)
    
    print('... Predicting output with best estimator')
    y_val_pred = cv_clf.predict(X_val_sc)
    get_classification_report(y_val, y_val_pred, sleep_states)
    get_feat_importances(cv_clf.best_estimator_, feat_cols)

    best_model = {'scl':scaler, 'smote_enn':smote_enn, 'clf':cv_clf.best_estimator_}
    pickle.dump(best_model, open(os.path.join(outdir,'balancing_smoteenn.pkl'),'wb'))   

    end_time = time.time()
    print('Time elapsed: %0.2fs' % (end_time-start_time))
    
    ############### Perform ML after balancing dataset with Balanced RF ###############
    
    print('\n... After balancing data with Balanced RF ...')
    start_time = time.time()
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_val_sc = scaler.transform(X_val)
    
    print('... Searching for suitable hyperparameters')
    clf = BalancedRandomForestClassifier(sampling_strategy='not majority', \
                                         class_weight='balanced_subsample', \
                                         n_estimators=100, max_depth=None, \
                                         random_state = 0)
    
    search_params = {'n_estimators':[50,100,150,200,250,300,500], \
                     'max_depth': [5,10,20,None]}
    cv_clf = RandomizedSearchCV(estimator=clf, param_distributions=search_params, \
                                cv=3, scoring='f1_macro', n_iter=10, n_jobs=-1)
    cv_clf.fit(X_train_sc, y_train)
    print(cv_clf.best_params_)

    print('... Predicting output with best estimator')
    y_val_pred = cv_clf.predict(X_val_sc)
    get_classification_report(y_val, y_val_pred, sleep_states)
    get_feat_importances(cv_clf.best_estimator_, feat_cols)

    best_model = {'scl':scaler, 'clf':cv_clf.best_estimator_}
    pickle.dump(best_model, open(os.path.join(outdir,'balanced_rf.pkl'),'wb'))   

    end_time = time.time()
    print('Time elapsed: %0.2fs' % (end_time-start_time))
    
if __name__ == "__main__":
    main(sys.argv[1:])
