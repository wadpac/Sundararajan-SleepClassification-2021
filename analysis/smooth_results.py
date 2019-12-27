import sys,os
import pandas as pd
import numpy as np

def main(argv):
  infile = argv[0]
  time_interval = int(argv[1])
  outfile = argv[2]

  df = pd.read_csv(infile)
  df = df.sort_values(['Fold','Users','Timestamp'])
  pred_cols = [col for col in df.columns if col.startswith('pred_')]
  users = list(set(df['Users']))
  sm_cols = ['smooth_' + col.split('pred_')[1] for col in pred_cols]
  for col in sm_cols:
    df[col] = 0.0
  for user in users:
    user_df = df[df['Users'] == user]  
    for col in pred_cols:
      sm_col = 'smooth_' + col.split('pred_')[1]
      timestamp = pd.to_datetime(user_df['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
      cls_df = pd.DataFrame(data={'timestamp':timestamp, 'cls_prob':user_df[col]})
      cls_df.set_index('timestamp', inplace=True)
      cls_mean = cls_df.rolling(str(time_interval)+'S').mean()
      df.loc[df['Users'] == user, sm_col] = cls_mean.values
  norm = df[sm_cols].sum(axis=1)
  for col in sm_cols:
    df[col] = df[col]/norm
  df.to_csv(outfile, index=False, sep=',')

if __name__ == "__main__":
  main(sys.argv[1:])
