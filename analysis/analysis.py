import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support, accuracy_score,\
                            classification_report, confusion_matrix,\
                            roc_auc_score, average_precision_score

def cv_save_classification_result(pred_list, sleep_states, fname, method='feat_eng'):
  # method is feature engineering (feat_eng) or deep learning (dl)
  nfolds = len(pred_list)
  for i in range(nfolds):
    if method == 'feat_eng': # Feature engineering 
      users = pred_list[i][0]
      timestamp = pred_list[i][1]
      fnames = pred_list[i][2]
      y_true = pred_list[i][3]
      y_pred = pred_list[i][4] # class probabilities
    else: # Deep learning
      users = pred_list[i][0]
      fnames = pred_list[i][1]
      y_true = pred_list[i][2]
      y_pred = pred_list[i][3] # class probabilities
    y_true_onehot = np.zeros((y_true.shape[0], len(sleep_states))) # convert to one-hot representation  
    y_true_onehot[np.arange(y_true.shape[0]), y_true] = 1
    fold = np.array([i+1]*y_true.shape[0])
    if method == 'feat_eng':
      df = pd.DataFrame({'Fold':fold, 'Users':users, 'Timestamp':timestamp, 'Filenames':fnames}).reset_index(drop=True)
    else:  
      df = pd.DataFrame({'Fold':fold, 'Users':users, 'Filenames':fnames}).reset_index(drop=True)
    true_cols = ['true_'+state for state in sleep_states]
    df_y_true = pd.DataFrame(y_true_onehot, columns=true_cols)
    pred_cols = ['pred_'+state for state in sleep_states]
    df_y_pred = pd.DataFrame(y_pred, columns=pred_cols)
    df = pd.concat([df, pd.concat([df_y_true, df_y_pred], axis=1)], axis=1)
    if i != 0:
      df.to_csv(fname, mode='a', header=False, index=False)  
    else:
      df.to_csv(fname, mode='w', header=True, index=False) 

def cv_save_feat_importances_result(importances, feature_names, fname):
  nfolds = len(importances)
  columns = ['Features'] + ['Fold'+str(fold+1) for fold in range(nfolds)]
  df_data = np.array([feature_names] + importances).T
  df = pd.DataFrame(df_data, columns=columns)
  df.to_csv(fname, mode='w', header=True, index=False)

def cv_get_feat_importances(fname):
  mean_importances = np.zeros(importances[0].shape)
  nfolds = len(importances)
  for i in range(nfolds):
    mean_importances = mean_importances + importances[i]
  mean_importances = mean_importances/nfolds
  indices = np.argsort(mean_importances)[::-1]
  print('Feature ranking:')
  for i in range(mean_importances.shape[0]):
    print('%d. %s: %0.4f' % (i+1,feature_names[indices[i]],mean_importances[indices[i]]))
 
def cv_get_classification_report(pred_list, mode, sleep_states, method='feat_eng'):
  # method is feature engineering (feat_eng) or deep learning (dl)
  nfolds = len(pred_list)
  nfolds = len(pred_list)
  precision = 0.0; recall = 0.0; fscore = 0.0; accuracy = 0.0
  class_metrics = {}
  for state in sleep_states:
      class_metrics[state] = {'precision':0.0, 'recall': 0.0, 'f1-score':0.0}
  confusion_mat = np.zeros((len(sleep_states),len(sleep_states)))
  sleep_labels = [idx for idx,state in enumerate(sleep_states)]
  for i in range(nfolds):
    if method == 'feat_eng':  
      y_true = pred_list[i][3]
      probs = pred_list[i][4]
    else:
      y_true = pred_list[i][2]
      probs = pred_list[i][3]
    y_pred = probs.argmax(axis=1)
    # Get metrics across all classes
    prec, rec, fsc, sup = precision_recall_fscore_support(y_true, y_pred,
                                                          average='macro')
    acc = accuracy_score(y_true, y_pred)
    precision += prec; recall += rec; fscore += fsc; accuracy += acc
    # Get metrics per class
    fold_class_metrics = classification_report(y_true, y_pred, labels=sleep_labels,
                                   target_names=sleep_states, output_dict=True)
    for state in sleep_states:
      class_metrics[state]['precision'] += fold_class_metrics[state]['precision']
      class_metrics[state]['recall'] += fold_class_metrics[state]['recall']
      class_metrics[state]['f1-score'] += fold_class_metrics[state]['f1-score']
    # Get confusion matrix
    fold_conf_mat = confusion_matrix(y_true, y_pred, labels=sleep_labels).astype(np.float)
    for idx,state in enumerate(sleep_states):
      fold_conf_mat[idx,:] = fold_conf_mat[idx,:] / float(len(y_true[y_true == sleep_labels[idx]]))
    confusion_mat = confusion_mat + fold_conf_mat

  # Average metrics across all folds
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
    print('%s\t\t%0.4f\t\t%0.4f\t\t%0.4f' % 
                      (state, class_metrics[state]['precision'],
                      class_metrics[state]['recall'], 
                      class_metrics[state]['f1-score']))
  print('\n')

  # Confusion matrix
  confusion_mat = confusion_mat / nfolds
  if mode == 'binary':
    print('ConfMat\tWake\tSleep\tNonwear\n')
    for i in range(confusion_mat.shape[0]):
      print('%s\t%0.4f\t%0.4f\t%0.4f' % 
               (sleep_states[i], confusion_mat[i][0],
                confusion_mat[i][1], confusion_mat[i][2]))
    print('\n')
  else:    
    print('ConfMat\tWake\tNREM1\tNREM2\tNREM3\tREM\tNonwear\n')
    for i in range(confusion_mat.shape[0]):
      print('%s\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f' % 
	       (sleep_states[i], confusion_mat[i][0], confusion_mat[i][1], 
	        confusion_mat[i][2], confusion_mat[i][3],
                confusion_mat[i][4], confusion_mat[i][5]))
    print('\n')

def cv_classification_report(infile, mode='binary'):
  df = pd.read_csv(infile)
  
  sleep_states = [col.split('_')[1] for col in df.columns if (col.startswith('true')) and (not col.endswith('Nonwear'))]
  sleep_labels = [idx for idx,state in enumerate(sleep_states)]
  true_cols = ['true_'+state for state in sleep_states]
  pred_cols = ['smooth_'+state for state in sleep_states]
  nclasses = len(true_cols)
  nfolds = len(set(df['Fold']))
  
  metrics = {'precision':0.0, 'recall': 0.0, 'f1-score':0.0, 'accuracy':0.0, 'AUC':0.0, 'AP':0.0}
  class_metrics = {}
  for state in (sleep_states + ['Nonwear']):
    class_metrics[state] = {'precision':0.0, 'recall': 0.0, 'f1-score':0.0, 'AUC':0.0, 'AP':0.0, 'samples':0.0}
  
  for fold in range(nfolds):  
    # Get overall scores excluding nonwear
    true_prob = df[df['Fold'] == fold+1][true_cols].values
    valid_indices = true_prob.sum(axis=1) > 0
    true_prob = true_prob[valid_indices]
    y_true = true_prob.argmax(axis=1)
    pred_prob = df[df['Fold'] == fold+1][pred_cols].values[valid_indices]
    pred_prob = pred_prob/pred_prob.sum(axis=1).reshape(-1,1)
    pred_prob[np.isnan(pred_prob)] = 1.0/nclasses
    y_pred = pred_prob.argmax(axis=1)
    prec, rec, fsc, sup = precision_recall_fscore_support(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(true_prob, pred_prob, average='macro', multi_class='ovr')
    ap = average_precision_score(true_prob, pred_prob, average='macro')
    metrics['precision'] += prec; metrics['recall'] += rec
    metrics['f1-score'] += fsc; metrics['accuracy'] += acc
    metrics['AUC'] += auc
    metrics['AP'] += ap

    # Get metrics per class except Nonwear
    fold_class_metrics = classification_report(y_true, y_pred, labels=sleep_labels,
                                   target_names=sleep_states, output_dict=True)
    for idx,state in enumerate(sleep_states):
      class_metrics[state]['precision'] += fold_class_metrics[state]['precision']
      class_metrics[state]['recall'] += fold_class_metrics[state]['recall']
      class_metrics[state]['f1-score'] += fold_class_metrics[state]['f1-score']
      class_metrics[state]['samples'] += fold_class_metrics[state]['support']
      auc = roc_auc_score(true_prob[:,idx], pred_prob[:,idx])
      class_metrics[state]['AUC'] += auc
      ap = average_precision_score(true_prob[:,idx], pred_prob[:,idx])
      class_metrics[state]['AP'] += ap
 
    # Get metrics for Nonwear
    true_prob = df[df['Fold'] == fold+1]['true_Nonwear'].values 
    pred_prob = 1.0 - df[df['Fold'] == fold+1][pred_cols].values.sum(axis=1)
    test_prob = pred_prob[true_prob > 0]
    thresh = 1.0/(nclasses+1)
    y_pred = pred_prob > thresh
    fold_class_metrics = classification_report(true_prob, y_pred, labels=[0,1], output_dict=True)
    class_metrics['Nonwear']['precision'] += fold_class_metrics['1']['precision']
    class_metrics['Nonwear']['recall'] += fold_class_metrics['1']['recall']
    class_metrics['Nonwear']['f1-score'] += fold_class_metrics['1']['f1-score']
    class_metrics['Nonwear']['samples'] += fold_class_metrics['1']['support']
    auc = roc_auc_score(true_prob, pred_prob)
    class_metrics['Nonwear']['AUC'] += auc
    ap = average_precision_score(true_prob, pred_prob)
    class_metrics['Nonwear']['AP'] += ap

  # Average metrics across all folds
  for key in metrics.keys():
    metrics[key] = metrics[key]/nfolds
    print('{} = {:0.4f}'.format(key, metrics[key]))

  # Classwise report
  sleep_states = [col.split('_')[1] for col in df.columns if col.startswith('true')]
  print('\nClass\t\tPrecision\tRecall\t\tF1-score\tAUC\t\tAP\t\tSamples')
  for state in sleep_states:
    class_metrics[state]['precision'] = class_metrics[state]['precision'] / nfolds
    class_metrics[state]['recall'] = class_metrics[state]['recall'] / nfolds
    class_metrics[state]['f1-score'] = class_metrics[state]['f1-score'] / nfolds
    class_metrics[state]['AUC'] = class_metrics[state]['AUC'] / nfolds
    class_metrics[state]['AP'] = class_metrics[state]['AP'] / nfolds
    print('%s\t\t%0.4f\t\t%0.4f\t\t%0.4f\t\t%0.4f\t\t%0.4f\t\t%d' % 
                      (state, class_metrics[state]['precision'],
                      class_metrics[state]['recall'], 
                      class_metrics[state]['f1-score'],
                      class_metrics[state]['AUC'],
                      class_metrics[state]['AP'],
                      class_metrics[state]['samples']))
  print('\n')
  
  ################################ Confusion matrix ##################################
  sleep_states = [col.split('_')[1] for col in df.columns if col.startswith('true')]
  sleep_labels = [idx for idx,state in enumerate(sleep_states)]
  true_cols = [col for col in df.columns if col.startswith('true')]
  pred_cols = [col for col in df.columns if col.startswith('smooth')]
  nclasses = len(true_cols)
  confusion_mat = np.zeros((len(sleep_states),len(sleep_states)))
  for fold in range(nfolds):
    true_prob = df[df['Fold'] == fold+1][true_cols].values  
    y_true = true_prob.argmax(axis=1)
    pred_prob = df[df['Fold'] == fold+1][pred_cols].values 
    y_pred = pred_prob.argmax(axis=1)
    fold_conf_mat = confusion_matrix(y_true, y_pred, labels=sleep_labels).astype(np.float)
    for idx,state in enumerate(sleep_states):
      fold_conf_mat[idx,:] = fold_conf_mat[idx,:] / float(len(y_true[y_true == sleep_labels[idx]]))
    confusion_mat = confusion_mat + fold_conf_mat

  confusion_mat = confusion_mat / nfolds
  if mode == 'binary':
    print('ConfMat\tWake\tSleep\tNonwear\n')
    for i in range(confusion_mat.shape[0]):
      print('%s\t%0.4f\t%0.4f\t%0.4f' % 
               (sleep_states[i], confusion_mat[i][0],
                confusion_mat[i][1], confusion_mat[i][2]))
    print('\n')
  else:    
    print('ConfMat\tWake\tNREM1\tNREM2\tNREM3\tREM\tNonwear\n')
    for i in range(confusion_mat.shape[0]):
      print('%s\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f' % 
	       (sleep_states[i], confusion_mat[i][0], confusion_mat[i][1], 
	        confusion_mat[i][2], confusion_mat[i][3],
                confusion_mat[i][4], confusion_mat[i][5]))
    print('\n')
