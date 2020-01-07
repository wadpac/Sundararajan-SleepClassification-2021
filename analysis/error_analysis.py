import sys,os
import argparse
import numpy as np
from scipy.signal import savgol_filter
import pandas as pd
from collections import Counter
from tqdm import tqdm
from entropy import spectral_entropy

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def get_hist(arr, bins=20):
  hist, edges = np.histogram(arr, bins=bins)
  hist = hist / hist.sum()
  hist = savgol_filter(hist, 25, 3)
  hist_bins = [(edges[i] + edges[i+1])/2.0 for i in range(bins)]
  return hist, hist_bins

def plot_hist(outdir, true_bins, true_hist, true_state, pred_bins, pred_hist, pred_state, metric='prob'):
  plt.plot(true_bins, true_hist, 'g-*', label='True-'+true_state)
  if true_state != pred_state:
    plt.plot(pred_bins, pred_hist, 'r-o', label='Pred-'+pred_state)
  plt.legend(loc='upper right')
  if metric == 'prob':
    plt.xlim([0,1])
  plt.savefig(os.path.join(args.outdir, metric+'_True-'+true_state+'_Pred-'+pred_state+'.jpg'))
  plt.close()

def main(args):
  if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

  df = pd.read_csv(args.result)
  states = [col.split('true_')[1] for col in df.columns if col.startswith('true')]
  true_cols = ['true_'+state for state in states]
  pred_cols = ['smooth_'+state for state in states]
  nclasses = len(true_cols)
  y_true = np.argmax(df[true_cols].values, axis=1)
  pred_prob = df[pred_cols].values
  y_pred = np.argmax(pred_prob, axis=1)
  indices = np.arange(y_true.shape[0])

  feat_df = pd.read_csv(os.path.join(args.indir, 'features_30.0s.csv'))
  shape_df = pd.read_csv(os.path.join(args.indir, 'datashape_30.0s.csv'))
  num_samples = shape_df['num_samples'].values[0]
  num_timesteps = shape_df['num_timesteps'].values[0]
  num_channels = shape_df['num_channels'].values[0]
  rawdata = np.memmap(os.path.join(args.indir, 'rawdata_30.0s.npz'), mode='r', dtype='float32',\
                      shape=(num_samples, num_timesteps, num_channels))
 
  # Get entropy of error scenarios
  spec_entropy = []
  for idx in tqdm(indices):
    sidx = feat_df[(df.iloc[idx]['Filenames'] == feat_df['filename']) & 
                   (df.iloc[idx]['Timestamp'] == feat_df['timestamp'])].index.values[0]
    enorm = np.sqrt(rawdata[sidx,:,0]**2 + rawdata[sidx,:,1]**2 + rawdata[sidx,:,2]**2)
    spec_entropy.append(spectral_entropy(enorm, 50, normalize=True))
  spec_entropy = np.array(spec_entropy) 
  spec_entropy[np.isnan(spec_entropy)] = 0.0

  # Plot probability distributions of true and pred states
  bins = 200
  for i,true_state in enumerate(states):
    for j,pred_state in enumerate(states):
      chosen_indices = indices[(y_true == i) & (y_pred == j)]
      state_true_prob = pred_prob[chosen_indices, i]
      state_pred_prob = pred_prob[chosen_indices, j]
      true_prob_hist, true_prob_bins = get_hist(state_true_prob, bins)
      pred_prob_hist, pred_prob_bins = get_hist(state_pred_prob, bins)
      plot_hist(args.outdir, true_prob_bins, true_prob_hist, true_state,\
                pred_prob_bins, pred_prob_hist, pred_state, metric='prob')

      state_true_ent = spec_entropy[(y_true == i) & (y_pred == i)]
      state_pred_ent = spec_entropy[(y_true == i) & (y_pred == j)]
      true_ent_hist, true_ent_bins = get_hist(state_true_ent, bins)
      pred_ent_hist, pred_ent_bins = get_hist(state_pred_ent, bins)
      plot_hist(args.outdir, true_ent_bins, true_ent_hist, true_state,\
                pred_ent_bins, pred_ent_hist, pred_state, metric='ent')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--result', type=str, help='Result file')
  parser.add_argument('--indir', type=str, help='Directory with input files')
  parser.add_argument('--outdir', type=str, help='Directory to store output plots')
  args = parser.parse_args()
  main(args)
