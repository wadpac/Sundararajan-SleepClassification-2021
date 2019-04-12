import sys,os
import numpy as np
import pandas as pd
import h5py
import math
from scipy.stats import entropy
from collections import Counter
import pickle

# Get Euclidean Norm minus One
def get_ENMO(x,y,z):
  enorm = np.sqrt(x*x + y*y + z*z)
  ENMO = np.maximum(enorm-1.0, 0.0)
  return ENMO

# Get tilt angles
def get_tilt_angles(x,y,z):
  angle_x = np.arctan2(x, np.sqrt(y*y + z*z)) * 180.0/math.pi
  angle_y = np.arctan2(y, np.sqrt(x*x + z*z)) * 180.0/math.pi
  angle_z = np.arctan2(z, np.sqrt(x*x + y*y)) * 180.0/math.pi
  return angle_x, angle_y, angle_z

# Get Locomotor Inactivity During Sleep   
def get_LIDS(timestamp, ENMO):
  df = pd.concat((timestamp, pd.Series(ENMO)), axis=1)
  df.columns = ['timestamp','ENMO']
  df.set_index('timestamp', inplace=True)
  
  df['ENMO_sub'] = np.where(ENMO < 0.02, 0, ENMO-0.02) # assuming ENMO is in g
  ENMO_sub_smooth = df['ENMO_sub'].rolling('600s').sum() # 10-minute rolling sum
  df['LIDS_unfiltered'] = 100.0 / (ENMO_sub_smooth + 1.0)
  LIDS = df['LIDS_unfiltered'].rolling('1800s').mean().values # 30-minute rolling average
  return LIDS

def compute_entropy(df, bins=20):
  hist, bin_edges = np.histogram(df, bins=bins)
  p = hist/float(hist.sum())
  ent = entropy(p)
  return ent

# Aggregate statistics of features over a given time interval
def get_stats(timestamp, feature, token_interval):
  feat_df = pd.DataFrame(data={'timestamp':timestamp, 'feature':feature})
  feat_df.set_index('timestamp', inplace=True)
  feat_mean = feat_df.resample(str(token_interval)+'S').mean()
  feat_std = feat_df.resample(str(token_interval)+'S').std()
  feat_min = feat_df.resample(str(token_interval)+'S').min()
  feat_max = feat_df.resample(str(token_interval)+'S').max()
  feat_mad = feat_df.resample(str(token_interval)+'S').apply(pd.DataFrame.mad)
  feat_ent1 = feat_df.resample(str(token_interval)+'S').apply(compute_entropy, bins=20)
  feat_ent2 = feat_df.resample(str(token_interval)+'S').apply(compute_entropy, bins=200)
  stats = np.vstack((feat_mean['feature'], feat_std['feature'], feat_min['feature'], 
                     feat_max['feature'], feat_mad['feature'], feat_ent1['feature'], 
                     feat_ent2['feature'])).T
  return stats

def get_categ(df, default='NaN'):
  ctr = Counter(df)
  for key in ctr:
    ctr[key] = ctr[key]/float(len(df))
  dom_categ = ctr.most_common()[0]
  if dom_categ[1] >= 0.7: # If a category occurs more than 70% of time interval, mark that as dominant category
    dom_categ = dom_categ[0]
  else:
    dom_categ = default
  return dom_categ
    
def get_dominant_categ(timestamp, categ, token_interval, default='NaN'):
  categ_df = pd.DataFrame(data={'timestamp':timestamp, 'category':categ})
  categ_df.set_index('timestamp', inplace=True)
  dom_categ = categ_df.resample(str(token_interval)+'S').apply(get_categ, default=default)
  return np.array(dom_categ['category'])   

# Get sequence labels in BIEO format - Beginning, Inside, End, Outside
def get_sequential_label(labels, nonwear, states):
  # Initialize all labels as 'O'
  seq_labels = ['O'] * len(labels)
  # Rename first and last labels of the sequence
  if labels[0] in states:
    seq_labels[0] = 'B-' + labels[0]
  if labels[-1] in states:
    seq_labels[-1] = 'E-' + labels[-1]
  # Rename all other labels based on their previous and next labels
  for i in range(1,len(labels)-1):
    # If nonwear, retain label as 'O'
    if nonwear[i] is True or labels[i] not in states:
      continue 
    # Label beginning of state
    if labels[i] != labels[i-1]:
      seq_labels[i] = 'B-' + labels[i]
    else: # Inside a state   
      seq_labels[i] = 'I-' + labels[i]
    # Label end of state
    if labels[i] != labels[i+1]:
      seq_labels[i] = 'E-' + labels[i]
  return seq_labels

def convert2seq(features, labels, n_seq_tokens=10, user=None, position=None, dataset=None):
  sequences = []
  ntokens = len(labels)
  columns = ['ENMO_mean','ENMO_std','ENMO_min','ENMO_max','ENMO_mad','ENMO_entropy1','ENMO_entropy2',
          'angz_mean','angz_std','angz_min','angz_max','angz_mad','angz_entropy1','angz_entropy2', 
          'LIDS_mean','LIDS_std','LIDS_min','LIDS_max','LIDS_mad','LIDS_entropy1','LIDS_entropy2']
  for st_idx in range(0,ntokens,n_seq_tokens):
    end_idx = min(ntokens, st_idx+n_seq_tokens)
    if (end_idx-st_idx) < (n_seq_tokens//2): # Discard last sequence if too short
      continue
    lbl_ctr = Counter(labels[st_idx:end_idx]).most_common()
    lbl_ctr = [(lbl,float(val)/n_seq_tokens) for lbl,val in lbl_ctr]
    # Discard sequences which are atleast 60% or more of 'O'
    if lbl_ctr[0][0] == 'O' and lbl_ctr[0][1] >= 0.6:
      continue
    else:
      feat_df = pd.DataFrame(features[st_idx:end_idx], columns=columns)
      feat = list(feat_df.T.to_dict().values())
      lbl = labels[st_idx:end_idx]
      sequences.append({'features': feat, 'labels': lbl, 'user': user, 'position': position, 'dataset': dataset})
  return sequences

def main(argv):
  indir = argv[0]
  token_interval = int(argv[1]) # Time interval in seconds for tokens in a seq
  num_seq_tokens = int(argv[2]) # Number of tokens in a sequence
  outdir = argv[3]

  outdir = os.path.join(outdir, 'seq_'+str(num_seq_tokens)+'tok_'+str(token_interval)+'sec') 
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  
  # Sleep states
  states = ['Wake','NREM 1','NREM 2','NREM 3','REM']
  
  files = os.listdir(indir)
  for idx,fname in enumerate(files):
    print('Processing ' + fname)
    
    fh = h5py.File(os.path.join(indir,fname), 'r')
    x = np.array(fh['X'])
    y = np.array(fh['Y'])
    z = np.array(fh['Z'])
    timestamp = pd.Series(fh['DateTime']).apply(lambda x: x.decode('utf8'))
    timestamp = pd.to_datetime(timestamp, format='%Y-%m-%d %H:%M:%S.%f')
        
    # Get ENMO and acceleration angles
    ENMO = get_ENMO(x,y,z)
    angle_x, angle_y, angle_z = get_tilt_angles(x,y,z)
    # Get LIDS (Locomotor Inactivity During Sleep)
    LIDS = get_LIDS(timestamp, ENMO)
        
    # Get statistics of features for given time intervals
    ENMO_stats = get_stats(timestamp, ENMO, token_interval)
    angle_z_stats = get_stats(timestamp, angle_z, token_interval)
    LIDS_stats = get_stats(timestamp, LIDS, token_interval)
    feat = np.hstack((ENMO_stats, angle_z_stats, LIDS_stats))
           
    # Get nonwear for each interval
    nonwear = np.array(fh['Nonwear'])
    nonwear_agg = get_dominant_categ(timestamp, nonwear, token_interval, default=True)
    
    # Standardize label names for both datasets
    # Get label for each interval
    label = np.array([x.decode('utf8') for x in np.array(fh['SleepState'])], dtype=object)
    label[label == 'W'] = 'Wake'
    label[label == 'N1'] = 'NREM 1'
    label[label == 'N2'] = 'NREM 2'
    label[label == 'N3'] = 'NREM 3'
    label[label == 'R'] = 'REM'
    label_agg = get_dominant_categ(timestamp, label, token_interval)
        
    # Get sequence labels for the user
    seq_label = get_sequential_label(label_agg, nonwear_agg, states)
   
    # Uncomment for PSGNewcastle2015 data
    user = fname.split('_')[0]
    position = fname.split('_')[1]
    dataset = 'Newcastle'        
#    # Uncomment for UPenn_Axivity data
#    user = fname.split('.h5')[0][-4:]
#    position = 'NaN'
#    dataset = 'UPenn'
        
    # Break up data into sequences of specified number of non-overlapping tokens
    # If over 70% of sequence is 'O', exclude that sequence 
    sequences = convert2seq(feat, seq_label, n_seq_tokens=num_seq_tokens, user=user, position=position, dataset=dataset)
    pickle.dump(sequences, open(os.path.join(outdir,fname.split('.h5')[0]+'.pkl'),'wb'))
    print(len(sequences))

if __name__ == '__main__':
  main(sys.argv[1:])
