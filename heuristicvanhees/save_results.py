import sys,os
import numpy as np
import pandas as pd
sys.path.append('../analysis/')
from analysis import cv_save_feat_importances_result, cv_save_classification_result

def main(argv):
  infile = argv[0]
  outdir = argv[1]

  sleep_states = ['Wake', 'Sleep', 'Nonwear']
  df = pd.read_csv(infile)
  df = df[df['label'] != 'Wake_ext'].reset_index()
  df['binary_label'] = df['label']
  df.loc[df['label'] == 'NREM 1','binary_label'] = 'Sleep'
  df.loc[df['label'] == 'NREM 2','binary_label'] = 'Sleep'
  df.loc[df['label'] == 'NREM 3','binary_label'] = 'Sleep'
  df.loc[df['label'] == 'REM','binary_label'] = 'Sleep'
  
  y_true = np.array([sleep_states.index(val) for val in df['binary_label']])
  y_pred = np.array([sleep_states.index(val) for val in df['heuristic']])
  y_pred_onehot = np.zeros((len(y_pred), len(sleep_states))) # convert to one-hot representation   
  y_pred_onehot[np.arange(len(y_pred)), y_pred] = 1
  predictions = [(df['user'], df['timestamp'], df['filename'], y_true, y_pred_onehot)]

  cv_save_classification_result(predictions, sleep_states, os.path.join(outdir, 'heuristic_classification.csv'))
 

if __name__ == "__main__":
  main(sys.argv[1:])
