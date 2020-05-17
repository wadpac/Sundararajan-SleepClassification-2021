import sys,os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

def main(argv):
  infile = argv[0]
  state = argv[1]

  df = pd.read_csv(infile)
  users = list(set(df['Users']))
  true_cols = [col for col in df.columns if col.startswith('true')]
  pred_cols = [col for col in df.columns if col.startswith('smooth')]

  sleep_perc = []
  fscore = []
  wake_fsc = []
  sleep_fsc = []
  for user in users:
    user_df = df[df['Users'] == user].reset_index(drop=True)
    if len(user_df):
      true_prob = user_df[true_cols].values
      y_true = true_prob.argmax(axis=1)
      pred_prob = user_df[pred_cols].values
      y_pred = pred_prob.argmax(axis=1)
      user_prec, user_rec, user_fsc, sup = precision_recall_fscore_support(y_true, y_pred, average='macro')
      perc = user_df['true_Sleep'].sum() / float(len(user_df))
      sleep_perc.append(perc)
      fscore.append(user_rec)
      class_metrics = classification_report(y_true, y_pred, labels=[0,1],
                                   target_names=['Wake','Sleep'], output_dict=True)
      wake_fsc.append(class_metrics['Wake']['f1-score'])
      sleep_fsc.append(class_metrics['Sleep']['f1-score'])

  wake_r, _ = spearmanr(sleep_perc, wake_fsc)
  sleep_r, _ = spearmanr(sleep_perc, sleep_fsc)
  rcoeff, _ = spearmanr(sleep_perc, fscore)
 
  plt.plot(sleep_perc, wake_fsc, 'g*', label='wake (r={:2f})'.format(wake_r))
  plt.plot(sleep_perc, sleep_fsc, 'ro', label='sleep (r={:2f})'.format(sleep_r))
  #plt.plot(sleep_perc, fscore, 'bs', label='healthy (r={:2f})'.format(rcoeff))
  plt.xlim([0,1]); plt.ylim([0,1])
  plt.xlabel('Time spent sleeping'); plt.ylabel('F-score')
  plt.legend(loc='lower left'); plt.title(state)
  plt.savefig(state+'.jpg')
  plt.close()

if __name__ == "__main__":
  main(sys.argv[1:])
