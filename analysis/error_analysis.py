import sys,os
import argparse
import numpy as np
import pandas as pd
from collections import Counter

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def main(args):
  if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
  df = pd.read_csv(args.dl_result)
  true_cols = [col for col in df.columns if col.startswith('true')]
  pred_cols = [col for col in df.columns if col.startswith('pred')]
  nclasses = len(true_cols)
  y_true = np.argmax(df[true_cols].values, axis=1)
  y_pred = np.argmax(df[pred_cols].values, axis=1)
  indices = np.arange(y_true.shape[0])
  error_indices = indices[y_true != y_pred]

  for i,idx in enumerate(error_indices):
    if y_true[idx] != 0:
      continue
    print('%d/%d' % (i+1,len(error_indices)))   
    fname = os.path.basename(df.iloc[idx]['Filenames'])
    X = np.load(os.path.join(args.indir,fname))
    xx = np.arange(X.shape[0])
    plt.Figure()
    plt.plot(xx, X[:,0], 'r', linewidth=2)
    plt.plot(xx, X[:,1], 'g', linewidth=2)
    plt.plot(xx, X[:,2], 'b', linewidth=2)
    plt.ylim([-1.5,1.5])
    plt.savefig(os.path.join(args.outdir, 'true' + str(y_true[idx]) + '_pred' + str(df[pred_cols].values[idx,y_true[idx]]) + '_' + fname.split('.npy')[0] + '.jpg'))
    plt.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dl_result', type=str, help='DL result file')
  parser.add_argument('--indir', type=str, help='Directory with input files')
  parser.add_argument('--outdir', type=str, help='Directory to store output plots')
  args = parser.parse_args()
  main(args)
