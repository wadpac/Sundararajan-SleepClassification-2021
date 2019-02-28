# -*- coding: utf-8 -*-

import sys,os
import pandas as pd
import numpy as np
import scipy
from collections import Counter

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def main(argv):
    infile = argv[0]
    
    # Read data file
    df = pd.read_csv(infile, dtype={'label':object, 'user':object, 'position':object, 'dataset':object})
    
    # Split data into train/validation/test based on users, not on samples
    # Use similar split up for Newcastle and UPenn datasets such that both 
    # are present in all partitions
    print('... Splitting data for hyperparameter search')
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
        
#    feat_cols = ['ENMO_mean','ENMO_std','ENMO_min','ENMO_max','ENMO_mad','ENMO_entropy1','ENMO_entropy2', \
#                 'angz_mean','angz_std','angz_min','angz_max','angz_mad','angz_entropy1','angz_entropy2', \
#                 'LIDS_mean','LIDS_std','LIDS_min','LIDS_max','LIDS_mad','LIDS_entropy1','LIDS_entropy2'
#]
    feat_cols = ['ENMO_mean','ENMO_std','ENMO_min','ENMO_max','ENMO_mad','ENMO_entropy', \
                 'angz_mean','angz_std','angz_min','angz_max','angz_mad','angz_entropy', \
                 'LIDS_mean','LIDS_std','LIDS_min','LIDS_max','LIDS_mad','LIDS_entropy'] 


    df_train = df[df['user'].isin(users_rest)].reset_index()
    X_train = df_train[feat_cols].values
    y_train = df_train['label'].values
    
    train_indices = df_train[df_train['user'].isin(users_train)].index
    val_indices = df_train[df_train['user'].isin(users_val)].index
    custom_cv = zip([train_indices],[val_indices])
    
    df_test = df[df['user'].isin(users_test)]
    X_test = df_test[feat_cols].values
    y_test = df_test['label'].values
    
    # Perform ML without balancing dataset
    # Perform grid search for hyper-parameter tuning to find best estimator using validation data
    # Fit training data to best estimator and compute metrics on test data
    svm = SVC()
    svm_parameters = {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1), 'kernel': ['rbf']}
    cv_clf = RandomizedSearchCV(estimator=svm, param_distributions=svm_parameters, cv=custom_cv, \
                                n_iter=10, n_jobs=10, scoring='f1_macro')
    print('... Searching for suitable hyperparameters')
    cv_clf.fit(X_train, y_train)
    print('... Predicting output with best estimator')
    y_test_pred = cv_clf.predict(X_test)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_test_pred, average='macro')
    accuracy = accuracy_score(y_test, y_test_pred)
    print('Precision = %0.4f' % (precision*100.0))
    print('Recall = %0.4f' % (recall*100.0))
    print('F-score = %0.4f' % (fscore*100.0))
    print('Accuracy = %0.4f' % (accuracy*100.0))
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
