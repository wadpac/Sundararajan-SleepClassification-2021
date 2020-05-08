import sys,os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.metrics import precision_recall_curve, average_precision_score
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

def main(argv):
  mode = argv[0]
  dataset = argv[1]
  outdir = argv[2]
  rf_file = argv[3]
  #dl_file = argv[4]
  input_files = [rf_file] #, dl_file]
  if mode == 'binary':
    heur_file = argv[4]
    input_files = [heur_file] + input_files
    states = ['Wake', 'Sleep']
    labels = ['vanHees', 'Random Forests'] #, 'Resnet']
    markers = ['r-o', 'b-d']#, 'g-*']
  else:
    if mode == 'multiclass':  
      states = ['Wake', 'NREM 1', 'NREM 2', 'NREM 3', 'REM']
    else:
      states = ['Wear', 'Nonwear']
    labels = ['Random Forests']#, 'Resnet']
    markers = ['b-d']#, 'g-*']
  curves = {state:[] for state in states}
  avg_prec = {state:[] for state in states}

  bins = 50
  for infile in input_files:  
    df = pd.read_csv(infile)
    nfolds = len(set(df['Fold']))
    true_cols = ['true_'+state for state in states]
    pred_cols = ['smooth_'+state for state in states]
    #if mode == 'nonwear': # want only a binary curve for nonwear detection
    #  pred_cols = ['smooth_'+state for state in (['Wear'] + states)]
    for state in states:
      y_true = df['true_'+state].values
      if mode == 'multiclass':
        pred_prob = 1.0 - df[[col for col in pred_cols if col != 'smooth_'+state]].sum(axis=1).values
      else:
        pred_prob = df['smooth_'+state].values
      precision, recall, th = precision_recall_curve(y_true, pred_prob)
      step = int(len(th)/bins) if (len(th) > 2*bins) else 1
      indices = np.arange(0, len(th), step)
      precision = precision[indices]; recall = recall[indices]
      ap = average_precision_score(y_true, pred_prob)
      curves[state].append(zip(recall*100.0, precision*100.0))
      avg_prec[state].append(ap*100.0)

  for state in states:
    for i,curve in enumerate(curves[state]):
      recall, precision = zip(*curve)
      plt.plot(recall, precision, markers[i], linewidth=2, label=labels[i] + ' (AP = {:0.2f}%)'.format(avg_prec[state][i]))
    plt.xlim([0,100]); plt.ylim([0,110])
    plt.xlabel('Recall (%)'); plt.ylabel('Precision (%)')
    if min(avg_prec[state]) > 30.0:
      plt.legend(loc='lower left', fontsize=12)
    else:
      plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, '_'.join(['PR',dataset, mode, state.replace(' ','')])+'.jpg'))
    plt.close()

if __name__ == "__main__":
  main(sys.argv[1:])
