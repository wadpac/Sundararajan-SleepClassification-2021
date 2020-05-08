import sys,os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

def main(argv):
  infile = argv[0]
  method = argv[1]
  mode = argv[2]
  dataset = argv[3]
  outdir = argv[4]

  df = pd.read_csv(infile)
  nfolds = len(set(df['Fold']))
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
  confusion_mat = confusion_mat*100.0 / nfolds

  # Plot confusion matrix
  plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
  plt.colorbar()
  tick_marks = np.arange(len(sleep_states))
  plt.xticks(tick_marks, sleep_states, rotation=45)
  plt.yticks(tick_marks, sleep_states)

  thresh = confusion_mat.max() / 2.0
  for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1])):
    plt.text(j, i, '{:0.2f}'.format(confusion_mat[i, j]),\
             fontsize=15, horizontalalignment="center",\
             color="white" if confusion_mat[i, j] > thresh else "black")

  plt.ylabel('True label', fontsize=18)
  plt.xlabel('Predicted label', fontsize=18)
  plt.tight_layout()
  plt.savefig(os.path.join(outdir, '-'.join((dataset, mode, 'confmat', method)) + '.jpg'))

if __name__ == "__main__":
  main(sys.argv[1:])
