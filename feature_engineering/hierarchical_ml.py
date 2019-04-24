import sys,os
import pandas as pd
from networkx import DiGraph
from collections import Counter

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import GroupKFold, StratifiedKFold

sys.path.append('../sklearn-hierarchical-classification/')
from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.metrics import h_fbeta_score, multi_labeled

from tqdm import tqdm
import networkx as nx

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def custom_h_fbeta(y_true, y_pred, graph=None):
    with multi_labeled(y_true, y_pred, graph) as (y_test_, y_pred_, graph_):
        h_fbeta = h_fbeta_score(
            y_test_,
            y_pred_,
            graph_,
        )
        return h_fbeta

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

    feat_cols = ['ENMO_mean','ENMO_std','ENMO_min','ENMO_max','ENMO_mad','ENMO_entropy1','ENMO_entropy2', 'ENMO_prevdiff', 'ENMO_nextdiff', \
                 'angz_mean','angz_std','angz_min','angz_max','angz_mad','angz_entropy1','angz_entropy2', 'angz_prevdiff', 'angz_nextdiff', \
                 'LIDS_mean','LIDS_std','LIDS_min','LIDS_max','LIDS_mad','LIDS_entropy1','LIDS_entropy2', 'LIDS_prevdiff', 'LIDS_nextdiff'
]
    X = df[feat_cols].values
    y = df['label']
    groups = df['user']

    # Class hierarchy for sleep stages
    class_hierarchy = {
        ROOT : {"Wake", "Sleep"},
        "Sleep" : {"NREM", "REM"},
        "NREM" : {"Light", "NREM 3"},
        "Light" : {"NREM 1", "NREM 2"} 
    }
    graph = DiGraph(class_hierarchy)
    
    outer_cv_splits = 5; inner_cv_splits = 3
    # Outer CV
    group_kfold = GroupKFold(n_splits=outer_cv_splits)
    out_fold = 0
    for train_indices, test_indices in group_kfold.split(X,y,groups):
        out_fold += 1
        print('Processing fold ' + str(out_fold))
        out_fold_X_train = X[train_indices,:]; out_fold_X_test = X[test_indices,:]
        out_fold_y_train = y[train_indices]; out_fold_y_test = y[test_indices]
        
        # Create a pipeline with scaler and hierarchical classifier
        pipe = Pipeline([('scaler', StandardScaler()),
                         ('clf', HierarchicalClassifier(
                            base_estimator=RandomForestClassifier(random_state=0, n_estimators=100, n_jobs=-1),
                            class_hierarchy=class_hierarchy,
                            prediction_depth='nmlnp',
                            progress_wrapper=tqdm,
                            stopping_criteria=0.7
                         ))
                        ])
        
        # Inner CV
        strat_kfold = StratifiedKFold(n_splits=inner_cv_splits, random_state=0, shuffle=True)       

        custom_cv_indices = []
        for grp_train_idx, grp_test_idx in strat_kfold.split(out_fold_X_train,out_fold_y_train):
            custom_cv_indices.append((grp_train_idx, grp_test_idx))
            
        print('Training')        
        search_params = {'clf__base_estimator__n_estimators':[50,100,200,300,500], \
             'clf__base_estimator__max_depth': [5,10,None]}
        cv_clf = RandomizedSearchCV(estimator=pipe, param_distributions=search_params, \
                                cv=custom_cv_indices, scoring=make_scorer(custom_h_fbeta,graph=graph), n_iter=5, \
                                n_jobs=-1, verbose=2)
        cv_clf.fit(out_fold_X_train, out_fold_y_train)
        print('Predicting')
        out_fold_y_pred = cv_clf.predict(out_fold_X_test)
    
        print("Classification Report:\n", classification_report(out_fold_y_test, out_fold_y_pred))
            
        # Demonstrate using our hierarchical metrics module with MLB wrapper
        with multi_labeled(out_fold_y_test, out_fold_y_pred, cv_clf.best_estimator_.named_steps['clf'].graph_) \
                                as (y_test_, y_pred_, graph_):
            h_fbeta = h_fbeta_score(
                y_test_,
                y_pred_,
                graph_,
            )
            print("h_fbeta_score: ", h_fbeta)

if __name__ == "__main__":
  main(sys.argv[1:])