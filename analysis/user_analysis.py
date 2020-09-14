import sys,os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report
from scipy.stats import spearmanr, ttest_ind

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(argv):
  infile = argv[0]
  userfile = argv[1]

  user_df = pd.read_csv(userfile)
  healthy_sleepers = user_df[user_df['sleep_disorder'] == 0]['user']
  poor_sleepers = user_df[user_df['sleep_disorder'] == 1]['user']
 
  df = pd.read_csv(infile)
  true_cols = [col for col in df.columns if col.startswith('true')]
  pred_cols = [col for col in df.columns if col.startswith('smooth')]

  # Healthy sleepers
  healthy_df = df[df['Users'].isin(healthy_sleepers)].reset_index(drop=True)
  healthy_sleep_perc = []
  healthy_fscore = []
  healthy_wake_fsc = []
  healthy_sleep_fsc = []
  for user in healthy_sleepers:
    user_df = healthy_df[healthy_df['Users'] == user].reset_index(drop=True)
    if len(user_df):
      true_prob = user_df[true_cols].values
      y_true = true_prob.argmax(axis=1)
      pred_prob = user_df[pred_cols].values
      y_pred = pred_prob.argmax(axis=1)
      user_prec, user_rec, user_fsc, sup = precision_recall_fscore_support(y_true, y_pred, average='macro')
      sleep_perc = user_df['true_Sleep'].sum() / float(len(user_df))
      healthy_sleep_perc.append(sleep_perc)
      healthy_fscore.append(user_rec)
      class_metrics = classification_report(y_true, y_pred, labels=[0,1],
                                   target_names=['Wake','Sleep'], output_dict=True)
      healthy_wake_fsc.append(class_metrics['Wake']['f1-score'])
      healthy_sleep_fsc.append(class_metrics['Sleep']['f1-score'])

  healthy_wake_r, _ = spearmanr(healthy_sleep_perc, healthy_wake_fsc)
  healthy_sleep_r, _ = spearmanr(healthy_sleep_perc, healthy_sleep_fsc)
  healthy_r, _ = spearmanr(healthy_sleep_perc, healthy_fscore)
 
  # Poor sleepers
  poor_df = df[df['Users'].isin(poor_sleepers)].reset_index(drop=True)
  poor_sleep_perc = []
  poor_fscore = []
  poor_wake_fsc = []
  poor_sleep_fsc = []
  for user in poor_sleepers:
    user_df = poor_df[poor_df['Users'] == user].reset_index(drop=True)
    if len(user_df):
      true_prob = user_df[true_cols].values
      y_true = true_prob.argmax(axis=1)
      pred_prob = user_df[pred_cols].values
      y_pred = pred_prob.argmax(axis=1)
      user_prec, user_rec, user_fsc, sup = precision_recall_fscore_support(y_true, y_pred, average='macro')
      sleep_perc = user_df['true_Sleep'].sum() / float(len(user_df))
      poor_sleep_perc.append(sleep_perc)
      poor_fscore.append(user_rec)
      class_metrics = classification_report(y_true, y_pred, labels=[0,1],
                                   target_names=['Wake','Sleep'], output_dict=True)
      poor_wake_fsc.append(class_metrics['Wake']['f1-score'])
      poor_sleep_fsc.append(class_metrics['Sleep']['f1-score'])
  
  poor_wake_r, _ = spearmanr(poor_sleep_perc, poor_wake_fsc)
  poor_sleep_r, _ = spearmanr(poor_sleep_perc, poor_sleep_fsc)
  poor_r, _ = spearmanr(poor_sleep_perc, poor_fscore)
 
  print(np.array(healthy_wake_fsc).mean()-np.array(poor_wake_fsc).mean())
  wake_t, wake_p = ttest_ind(healthy_wake_fsc, poor_wake_fsc, equal_var=False)
  print('Wake p = {:0.4f}'.format(wake_p))
  print(np.array(healthy_sleep_fsc).mean() - np.array(poor_sleep_fsc).mean())
  sleep_t, sleep_p = ttest_ind(healthy_sleep_fsc, poor_sleep_fsc, equal_var=False)
  print('Sleep p = {:0.4f}'.format(sleep_p))

  plt.plot(healthy_sleep_perc, healthy_wake_fsc, 'g*', label='wake (r={:2f})'.format(healthy_wake_r))
  plt.plot(healthy_sleep_perc, healthy_sleep_fsc, 'ro', label='sleep (r={:2f})'.format(healthy_sleep_r))
  plt.plot(healthy_sleep_perc, healthy_fscore, 'bs', label='healthy (r={:2f})'.format(healthy_r))
  plt.xlim([0,1]); plt.ylim([0,1])
  plt.xlabel('Time spent sleeping'); plt.ylabel('F-score')
  plt.legend(loc='lower left'); plt.title('Healthy')
  plt.savefig('healthy.jpg')
  plt.close()
  plt.plot(poor_sleep_perc, poor_wake_fsc, 'g*', label='wake (r={:.2f})'.format(poor_wake_r))
  plt.plot(poor_sleep_perc, poor_sleep_fsc, 'ro', label='sleep (r={:.2f})'.format(poor_sleep_r))
  plt.plot(poor_sleep_perc, poor_fscore, 'bs', label='poor (r={:.2f})'.format(poor_r))
  plt.xlim([0,1]); plt.ylim([0,1])
  plt.xlabel('Time spent sleeping'); plt.ylabel('F-score')
  plt.legend(loc='lower left'); plt.title('Poor')
  plt.savefig('poor.jpg')
  plt.close()

if __name__ == "__main__":
  main(sys.argv[1:])
