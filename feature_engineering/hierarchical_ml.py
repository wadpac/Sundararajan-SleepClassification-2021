import sys,os
import pandas as pd
import numpy as np
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
from sklearn_hierarchical_classification.metrics import h_fbeta_score, multi_labeled, fill_ancestors

from tqdm import tqdm
import networkx as nx
from networkx import DiGraph
from networkx.drawing.nx_agraph import graphviz_layout

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_classification_report(results):
    sleep_states = ['Wake','Sleep','REM','NREM','NREM 3','Light','NREM 1','NREM 2','Overall']
    
    print('\nState\t\tPrecision\tRecall\t\tFbeta\n')
    for key in sleep_states:
        precision = np.array(results[key]['precision']).mean()
        recall = np.array(results[key]['recall']).mean()
        fbeta = np.array(results[key]['fbeta']).mean()
        print('%s\t\t%0.4f\t\t%0.4f\t\t%0.4f' % (key,precision,recall,fbeta))  
    print('\n')

def custom_h_fbeta(y_true, y_pred, graph=None):
    with multi_labeled(y_true, y_pred, graph) as (y_test_, y_pred_, graph_, classes_):
        h_prec, h_rec, h_fbeta = h_fbeta_score(
            y_test_,
            y_pred_,
            graph_,
        )
        return h_fbeta
    
def get_node_metrics(y_true, y_pred, classes, node, beta=1.0):
    class_idx = classes.index(node)
    pred_pos = y_pred[:,class_idx]
    pos = y_true[:,class_idx]
    tp = pred_pos & pos
    prec = tp.sum()/float(pred_pos.sum())
    rec = tp.sum()/float(pos.sum())
    fbeta = (1. + beta ** 2.) * prec * rec / (beta ** 2. * prec + rec)
    support = pos.sum()
    return prec, rec, fbeta, support

def main(argv):
    infile = argv[0]
  
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
    factor = 10.0
    
    results = {'Wake': {'precision': [], 'recall': [], 'fbeta': []},
               'Sleep': {'precision': [], 'recall': [], 'fbeta': []},
               'REM': {'precision': [], 'recall': [], 'fbeta': []},
               'NREM': {'precision': [], 'recall': [], 'fbeta': []},
               'NREM 3': {'precision': [], 'recall': [], 'fbeta': []},
               'Light': {'precision': [], 'recall': [], 'fbeta': []},
               'NREM 1': {'precision': [], 'recall': [], 'fbeta': []},
               'NREM 2': {'precision': [], 'recall': [], 'fbeta': []},
               'Overall': {'precision': [], 'recall': [], 'fbeta': []}
              }   
    
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
                            prediction_depth='mlnp',
                            progress_wrapper=tqdm,
                            #stopping_criteria=0.7
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
                           n_jobs=-1, verbose=1)
        cv_clf.fit(out_fold_X_train, out_fold_y_train)
        print('Predicting')
        out_fold_y_pred = cv_clf.predict(out_fold_X_test)
        
        best_clf = cv_clf.best_estimator_
            
        # Demonstrate using our hierarchical metrics module with MLB wrapper
        with multi_labeled(out_fold_y_test, out_fold_y_pred, best_clf.named_steps['clf'].graph_) \
                                as (y_test_, y_pred_, graph_, classes_):
            fold_h_prec, fold_h_rec, fold_h_fbeta = h_fbeta_score(
                y_test_,
                y_pred_,
                graph_,
            )
            results['Overall']['precision'].append(fold_h_prec); results['Overall']['recall'].append(fold_h_rec)
            results['Overall']['fbeta'].append(fold_h_fbeta)
            print("Fold %d: precision: %0.4f, recall: %0.4f, fbeta: %0.4f" % (out_fold, fold_h_prec, fold_h_rec, fold_h_fbeta))
            
            y_test_ = fill_ancestors(y_test_, graph=graph_)
            y_pred_ = fill_ancestors(y_pred_, graph=graph_)
    
            fold_wake_prec, fold_wake_rec, fold_wake_fbeta, _ = get_node_metrics(y_test_, y_pred_, classes_, 'Wake')
            fold_sleep_prec, fold_sleep_rec, fold_sleep_fbeta, _ = get_node_metrics(y_test_, y_pred_, classes_, 'Sleep')
            fold_rem_prec, fold_rem_rec, fold_rem_fbeta, _ = get_node_metrics(y_test_, y_pred_, classes_, 'REM')
            fold_nrem_prec, fold_nrem_rec, fold_nrem_fbeta, _ = get_node_metrics(y_test_, y_pred_, classes_, 'NREM')
            fold_nrem3_prec, fold_nrem3_rec, fold_nrem3_fbeta, _ = get_node_metrics(y_test_, y_pred_, classes_, 'NREM 3')
            fold_light_prec, fold_light_rec, fold_light_fbeta, _ = get_node_metrics(y_test_, y_pred_, classes_, 'Light')
            fold_nrem1_prec, fold_nrem1_rec, fold_nrem1_fbeta, _ = get_node_metrics(y_test_, y_pred_, classes_, 'NREM 1')
            fold_nrem2_prec, fold_nrem2_rec, fold_nrem2_fbeta, _ = get_node_metrics(y_test_, y_pred_, classes_, 'NREM 2')
            
            results['Wake']['precision'].append(fold_wake_prec); results['Wake']['recall'].append(fold_wake_rec)
            results['Wake']['fbeta'].append(fold_wake_fbeta) 
            results['Sleep']['precision'].append(fold_sleep_prec); results['Sleep']['recall'].append(fold_sleep_rec)
            results['Sleep']['fbeta'].append(fold_sleep_fbeta) 
            results['REM']['precision'].append(fold_rem_prec); results['REM']['recall'].append(fold_rem_rec)
            results['REM']['fbeta'].append(fold_rem_fbeta) 
            results['NREM']['precision'].append(fold_nrem_prec); results['NREM']['recall'].append(fold_nrem_rec)
            results['NREM']['fbeta'].append(fold_nrem_fbeta) 
            results['NREM 3']['precision'].append(fold_nrem3_prec); results['NREM 3']['recall'].append(fold_nrem3_rec)
            results['NREM 3']['fbeta'].append(fold_nrem3_fbeta) 
            results['Light']['precision'].append(fold_light_prec); results['Light']['recall'].append(fold_light_rec)
            results['Light']['fbeta'].append(fold_light_fbeta) 
            results['NREM 1']['precision'].append(fold_nrem1_prec); results['NREM 1']['recall'].append(fold_nrem1_rec)
            results['NREM 1']['fbeta'].append(fold_nrem1_fbeta) 
            results['NREM 2']['precision'].append(fold_nrem2_prec); results['NREM 2']['recall'].append(fold_nrem2_rec)
            results['NREM 2']['fbeta'].append(fold_nrem2_fbeta) 
                        
#            # Plot graph importance
#            G = DiGraph(class_hierarchy)
#            G.add_edge('<ROOT>','Wake',weight=factor*fold_wake_fbeta)
#            G.add_edge('<ROOT>','Sleep',weight=factor*fold_sleep_fbeta)
#            G.add_edge('Sleep','REM',weight=factor*fold_rem_fbeta)
#            G.add_edge('Sleep','NREM',weight=factor*fold_nrem_fbeta)
#            G.add_edge('NREM','NREM 3',weight=factor*fold_nrem3_fbeta)
#            G.add_edge('NREM','Light',weight=factor*fold_light_fbeta)
#            G.add_edge('Light','NREM 1',weight=factor*fold_nrem1_fbeta)
#            G.add_edge('Light','NREM 2',weight=factor*fold_nrem2_fbeta)
#        
#            pos = graphviz_layout(G, prog='sfdp')
#    
#            edges = G.edges
#            weights = [wt for u,v,wt in G.edges.data('weight', default=1)]
#    
#            nx.draw(G, pos, edges=edges, width=weights, node_size=2000, with_labels=True)
#            
#            plt.savefig("plots/graph_fold"+str(out_fold)+".jpg")
            
    get_classification_report(results)

if __name__ == "__main__":
  main(sys.argv[1:])
