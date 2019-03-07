# -*- coding: utf-8 -*-

import sys,os
import pandas as pd
import numpy as np
import scipy
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn import manifold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def main(argv):
    infile = argv[0]
    
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

#    # Plot the embedded features
#    print('... Plotting embedding')
#    colors = ['blue','red','green','cyan','black']
#    X = df[feat_cols].values
#    Y = df['label'].values
#    tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=200.0) 
#    Xt = tsne.fit_transform(X)
#    plt.figure()
#    sns.scatterplot(x=Xt[:,0], y=Xt[:,1], hue=Y)
#    plt.title('TSNE embedding of data')
#    plt.xlabel('Dim1')
#    plt.ylabel('Dim2')
#    plt.savefig('TSNE-embedding.jpg')

    # Split data into train/validation/test based on users, not on samples
    # Use similar split up for Newcastle and UPenn datasets such that both 
    # are present in all partitions
    print('... Partitioning data')
    users1 = list(set(df.loc[df['dataset'] == 'Newcastle','user']))
    users1_rest, users1_test = train_test_split(users1, test_size=0.3, random_state=42)
    users1_train, users1_val = train_test_split(users1_rest, test_size=0.2, random_state=42)
    users2 = list(set(df.loc[df['dataset'] == 'UPenn','user']))
    users2_rest, users2_test = train_test_split(users2, test_size=0.3, random_state=42)
    users2_train, users2_val = train_test_split(users2_rest, test_size=0.2, random_state=42)
    users = users1 + users2
    users_rest = users1_rest + users2_rest
    users_train = users1_train + users2_train
    users_val = users1_val + users2_val
    users_test = users1_test + users2_test

    df_train_val = df[df['user'].isin(users_rest)].reset_index()
    X_train_val = df_train_val[feat_cols].values
    y_train_val = df_train_val['label'].values
    y_train_val = np.array([sleep_states.index(y) for y in y_train_val])
    
    train_indices = df_train_val[df_train_val['user'].isin(users_train)].index
    val_indices = df_train_val[df_train_val['user'].isin(users_val)].index
    custom_cv = zip([train_indices],[val_indices])
    X_train = X_train_val[train_indices,:]; y_train = y_train_val[train_indices]
    X_val = X_train_val[val_indices,:]; y_val = y_train_val[val_indices]
    
    df_test = df[df['user'].isin(users_test)]
    X_test = df_test[feat_cols].values
    y_test = df_test['label'].values
    y_test = np.array([sleep_states.index(y) for y in y_test])

    ################## Perform ML without balancing dataset #####################
    
    print('... Without balancing data ...')
    y_train_str = [sleep_states[y] for y in y_train]
    print(Counter(y_train_str))
    # Perform data normalization
    # Compute mean and std only using training data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_val_sc = scaler.transform(X_train_val)
    X_test_sc = scaler.transform(X_test)
    
    # Perform random search for hyper-parameter tuning to find best estimator using validation data
    # Fit training data to best estimator and compute metrics on test data
    print('... Searching for suitable hyperparameters')
    clf_params = {'n_estimators':[50,100,150,200,250,300,500]}
    clf = RandomForestClassifier(class_weight='balanced', random_state=0)
    cv_clf = RandomizedSearchCV(estimator=clf, param_distributions=clf_params, \
                                cv=custom_cv, scoring='f1_macro', \
                                n_iter=10, n_jobs=-1)
    cv_clf.fit(X_train_val_sc, y_train_val)
    print(cv_clf.best_params_)
    print('... Predicting output with best estimator')
    y_test_pred = cv_clf.predict(X_test_sc)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_test_pred, average='macro')
    accuracy = accuracy_score(y_test, y_test_pred)
    print('Precision = %0.4f' % (precision*100.0))
    print('Recall = %0.4f' % (recall*100.0))
    print('F-score = %0.4f' % (fscore*100.0))
    print('Accuracy = %0.4f' % (accuracy*100.0))
    print(classification_report(y_test, y_test_pred, target_names=sleep_states)) 
    
    ############### Perform ML after balancing dataset with SMOTE ###############
    
    print('... After balancing data with SMOTEENN ...')
    # Resample training data
    smote_enn = SMOTEENN(random_state=0, sampling_strategy='not majority')
    X_train_resamp, y_train_resamp = smote_enn.fit_resample(X_train, y_train)
    y_train_resamp_str = [sleep_states[y] for y in y_train_resamp]
    print(Counter(y_train_resamp_str))

    # Perform data normalization
    # Compute mean and std only using training data
    scaler = StandardScaler()
    scaler.fit(X_train_resamp)
    X_train_resamp_sc = scaler.transform(X_train_resamp)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)
  
    X_train_resamp_val_sc = np.vstack((X_train_resamp_sc, X_val_sc))
    y_train_resamp_val = np.concatenate((y_train_resamp, y_val), axis=None)
    ntrain = X_train_resamp.shape[0]; nval = X_val.shape[0]
    train_indices_resamp = list(range(ntrain))
    val_indices_resamp = list(range(ntrain, ntrain+nval)) 
    custom_cv = zip([train_indices_resamp],[val_indices_resamp])
    
    print('... Searching for suitable hyperparameters')
    clf = RandomForestClassifier(class_weight='balanced', random_state=0)
    cv_clf = RandomizedSearchCV(estimator=clf, param_distributions=clf_params, \
                                cv=custom_cv, scoring='f1_macro', \
                                n_iter=10, n_jobs=-1)
    cv_clf.fit(X_train_resamp_val_sc, y_train_resamp_val)
    print(cv_clf.best_params_)
    print('... Predicting output with best estimator')
    y_test_pred = cv_clf.predict(X_test_sc)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_test_pred, average='macro')
    accuracy = accuracy_score(y_test, y_test_pred)
    print('Precision = %0.4f' % (precision*100.0))
    print('Recall = %0.4f' % (recall*100.0))
    print('F-score = %0.4f' % (fscore*100.0))
    print('Accuracy = %0.4f' % (accuracy*100.0))
    print(classification_report(y_test, y_test_pred, target_names=sleep_states)) 
    
    ############### Perform ML after balancing dataset with Balanced RF ###############
    
    print('... After balancing data with Balanced RF ...')
    
    # Perform data normalization
    # Compute mean and std only using training data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_val_sc = scaler.transform(X_train_val)
    X_test_sc = scaler.transform(X_test)
  
    print('... Searching for suitable hyperparameters')
    clf = BalancedRandomForestClassifier(sampling_strategy='all', class_weight='balanced', \
                                         random_state = 0)
    custom_cv = zip([train_indices],[val_indices])
    cv_clf = RandomizedSearchCV(estimator=clf, param_distributions=clf_params, \
                                cv=custom_cv, scoring='f1_macro', \
                                n_iter=10, n_jobs=-1)
    cv_clf.fit(X_train_val_sc, y_train_val)
    print(cv_clf.best_params_)
    print('... Predicting output with best estimator')
    y_test_pred = cv_clf.predict(X_test_sc)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_test_pred, average='macro')
    accuracy = accuracy_score(y_test, y_test_pred)
    print('Precision = %0.4f' % (precision*100.0))
    print('Recall = %0.4f' % (recall*100.0))
    print('F-score = %0.4f' % (fscore*100.0))
    print('Accuracy = %0.4f' % (accuracy*100.0))
    print(classification_report(y_test, y_test_pred, target_names=sleep_states)) 
    
if __name__ == "__main__":
    main(sys.argv[1:])
