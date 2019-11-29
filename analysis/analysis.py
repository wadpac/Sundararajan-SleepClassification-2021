import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score,\
                            classification_report, confusion_matrix

def cv_save_classification_result(pred_list, sleep_states, fname):
  nfolds = len(pred_list)
  for i in range(nfolds):
    users = pred_list[i][0]
    timestamp = pred_list[i][1]
    fnames = pred_list[i][2]
    y_true = pred_list[i][3]
    y_true_onehot = np.zeros((y_true.shape[0], len(sleep_states))) # convert to one-hot representation  
    y_true_onehot[np.arange(y_true.shape[0]), y_true] = 1
    y_pred = pred_list[i][4] # class probabilities
    fold = np.array([i+1]*users.shape[0])
    df = pd.DataFrame({'Fold':fold, 'Users':users, 'Timestamp':timestamp, 'Filenames':fnames}).reset_index(drop=True)
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
 
def cv_get_classification_report(pred_list, mode, sleep_states):
  nfolds = len(pred_list)
  precision = 0.0; recall = 0.0; fscore = 0.0; accuracy = 0.0
  class_metrics = {}
  for state in sleep_states:
    class_metrics[state] = {'precision':0.0, 'recall': 0.0, 'f1-score':0.0}
  confusion_mat = np.zeros((len(sleep_states),len(sleep_states)))
  sleep_labels = [idx for idx,state in enumerate(sleep_states)]
  for i in range(nfolds):
    y_true = pred_list[i][2]
    y_pred = pred_list[i][3]
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
