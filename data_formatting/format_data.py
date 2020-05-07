# -*- coding: utf-8 -*-
import sys,os
import h5py
import numpy as np
import math
import pandas as pd
from scipy.stats import entropy
from collections import Counter

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
  # 10-minute rolling sum
  ENMO_sub_smooth = df['ENMO_sub'].rolling('600s').sum()
  df['LIDS_unfiltered'] = 100.0 / (ENMO_sub_smooth + 1.0)
  # 30-minute rolling average
  LIDS = df['LIDS_unfiltered'].rolling('1800s').mean().values
  return LIDS
  
def mad(data):
  out = np.mean(np.absolute(data - np.mean(data)))
  return out  

def compute_entropy(data, bins=20):
  bins = int(bins)
  hist, bin_edges = np.histogram(data, bins=bins)
  p = hist/float(hist.sum())
  ent = entropy(p)
  return ent 

# Get difference of feature with respect to prev or next interval
def get_diff_feat(feature, direction, time_diff, time_interval=30):
  window = int(time_diff / float(time_interval))
  mean_feature = feature.rolling(window).mean().values
  feature = feature.values.reshape(-1,1)
  mean_feature = mean_feature.reshape(-1,1)
  diff = np.zeros(feature.shape)
  if direction == 'prev':
    # Compute mean for border conditions
    for i in range(0,window-1):
      mean_feature[i] = np.mean(feature[:i+1])
  else:
    # Shift mean feature by window
    mean_feature = np.vstack((mean_feature[window-1:].reshape(-1,1),\
                              mean_feature[:window-1].reshape(-1,1)))
    # Compute mean for border conditions
    for i in range(len(feature)-window+1,len(feature)):
      mean_feature[i] = np.mean(feature[i:])
  if direction == 'prev':
    diff[1:] = feature[1:] - mean_feature[:-1]
  else:
    diff[:-1] = mean_feature[1:] - feature[:-1]

  return diff.reshape(-1,)
    
# Aggregate statistics of features over a given time interval
def get_stats(timestamp, feature, time_interval):
  feat_df = pd.DataFrame(data={'timestamp':timestamp, 'feature':feature})
  feat_df.set_index('timestamp', inplace=True)
  feat_mean = feat_df.resample(str(time_interval)+'S').mean()
  feat_std = feat_df.resample(str(time_interval)+'S').std()
  feat_min = feat_df.resample(str(time_interval)+'S').min()
  feat_max = feat_df.resample(str(time_interval)+'S').max()
  feat_range = feat_max['feature'] - feat_min['feature']
  feat_mad = feat_df.resample(str(time_interval)+'S').apply(mad)
  feat_ent1 = feat_df.resample(str(time_interval)+'S')\
                              .apply(compute_entropy, bins=20)
  feat_ent2 = feat_df.resample(str(time_interval)+'S')\
                              .apply(compute_entropy, bins=200)
  feat_prev30diff = get_diff_feat(feat_mean['feature'], 'prev', 30, time_interval)
  feat_next30diff = get_diff_feat(feat_mean['feature'], 'next', 30, time_interval)
  feat_prev60diff = get_diff_feat(feat_mean['feature'], 'prev', 60, time_interval)
  feat_next60diff = get_diff_feat(feat_mean['feature'], 'next', 60, time_interval)
  feat_prev120diff = get_diff_feat(feat_mean['feature'], 'prev', 120, time_interval)
  feat_next120diff = get_diff_feat(feat_mean['feature'], 'next', 120, time_interval)
  stats = np.vstack((feat_mean['feature'], feat_std['feature'], feat_range, 
                     feat_mad['feature'], feat_ent1['feature'], feat_ent2['feature'],
                     feat_prev30diff, feat_next30diff, feat_prev60diff, 
                     feat_next60diff, feat_prev120diff, feat_next120diff)).T

  # return index and stats
  return np.array(feat_mean.index.values, dtype='str'), stats

def get_categ(df, default='NaN'):
  ctr = Counter(df)
  for key in ctr:
    ctr[key] = ctr[key]/float(len(df))
  dom_categ = ctr.most_common()[0]
  # If a category occurs more than 70% of time interval,
  # mark that as dominant category
  if dom_categ[1] >= 0.7: 
    dom_categ = dom_categ[0]
  else:
    dom_categ = default
  return dom_categ
    
def get_dominant_categ(timestamp, categ, time_interval, default='NaN'):
  categ_df = pd.DataFrame(data={'timestamp':timestamp, 'category':categ})
  categ_df.set_index('timestamp', inplace=True)
  dom_categ = categ_df.resample(str(time_interval)+'S')\
                               .apply(get_categ, default=default)
  return np.array(dom_categ['category'])   

def get_tslice(df):
  tslice = np.array(df['channel'])
  return tslice

def get_timeslices(timestamp, channel, time_interval):
  chan_df = pd.DataFrame(data={'timestamp':timestamp, 'channel':channel})
  chan_df.set_index('timestamp', inplace=True)
  chan_slices = chan_df.resample(str(time_interval)+'S').apply(get_tslice)
  lengths = [tslice.shape[0] for tslice in chan_slices]
  resize_len = int(np.median(lengths))
  for tslice in chan_slices:
    tslice.resize((resize_len), refcheck=False) 
  return np.stack(chan_slices)

def resample_timeslices(data, num_timesteps):
  # Get resampled timesteps 
  tt = np.zeros((num_timesteps,), dtype=int)
  tt[1:-1] = np.sort(np.random.randint(1,data.shape[1]-1,num_timesteps-2))
  tt[-1] = data.shape[1]-1
  # Resample data
  num_channels = data.shape[2]
  resamp_data = np.zeros((data.shape[0], tt.shape[0], data.shape[2]))
  for i in range(data.shape[0]):
    for ch in range(num_channels):  
      resamp_data[i,:,ch] = np.interp(tt, np.arange(data.shape[1]), data[i,:,ch])
  return resamp_data

def process_file(fname, time_interval, sleep_states, dataset, num_timesteps):
  fh = h5py.File(fname, 'r')
  filename = os.path.basename(fname)

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
  
  # Get nonwear for each interval
  nonwear = np.array(fh['Nonwear'])
  nonwear_agg = get_dominant_categ(timestamp, nonwear, time_interval, default=True)
  
  # Standardize label names for both datasets
  # Get label for each interval
  label = np.array([x.decode('utf8') for x in np.array(fh['SleepState'])],
                   dtype=object)
  label[label == 'W'] = 'Wake'
  label[label == 'N1'] = 'NREM 1'
  label[label == 'N2'] = 'NREM 2'
  label[label == 'N3'] = 'NREM 3'
  label[label == 'R'] = 'REM'
  label[label == 'Wakefulness'] = 'Wake'
  label_agg = get_dominant_categ(timestamp, label, time_interval)
  label_agg[(np.isin(label_agg, sleep_states, invert=True))
            & (nonwear_agg == True)] = 'Nonwear'

  #################  Get features  ######################

  # Get statistics of features for given time intervals
  _, ENMO_stats = get_stats(timestamp, ENMO, time_interval)
  _, angle_z_stats = get_stats(timestamp, angle_z, time_interval)
  timestamp_agg, LIDS_stats = get_stats(timestamp, LIDS, time_interval)
  feat = np.hstack((ENMO_stats, angle_z_stats, LIDS_stats))

  # Get valid timestamps, features and labels
  timestamp_valid = timestamp_agg[np.isin(label_agg, sleep_states)].reshape(-1,1)
  feat_valid = feat[np.isin(label_agg, sleep_states),:]
  label_valid = label_agg[np.isin(label_agg, sleep_states)].reshape(-1,1)
    
  # Write features to CSV file
  data = np.hstack((timestamp_valid.reshape(-1,1), feat_valid, label_valid.reshape(-1,1)))
  cols = ['timestamp','ENMO_mean','ENMO_std','ENMO_range','ENMO_mad',
          'ENMO_entropy1','ENMO_entropy2','ENMO_prev30diff','ENMO_next30diff',
          'ENMO_prev60diff', 'ENMO_next60diff', 'ENMO_prev120diff', 'ENMO_next120diff',  
          'angz_mean','angz_std','angz_range','angz_mad',
          'angz_entropy1','angz_entropy2','angz_prev30diff','angz_next30diff', 
          'angz_prev60diff', 'angz_next60diff', 'angz_prev120diff', 'angz_next120diff',  
          'LIDS_mean','LIDS_std','LIDS_range','LIDS_mad',
          'LIDS_entropy1','LIDS_entropy2','LIDS_prev30diff','LIDS_next30diff',
          'LIDS_prev60diff', 'LIDS_next60diff', 'LIDS_prev120diff', 'LIDS_next120diff', 'label']
  df = pd.DataFrame(data=data, columns=cols)
  
  if dataset == 'Newcastle':
    user = filename.split('_')[0]
    position = filename.split('_')[1]
    dataset = 'Newcastle'        
  elif dataset == 'UPenn':
    user = filename.split('.h5')[0][-4:]
    position = 'NaN'
    dataset = 'UPenn'
  elif dataset == 'AMC':
    user = '_'.join(part for part in filename.split('.h5')[0].split('_')[0:2])
    position = 'NaN'
    dataset = 'AMC'

  df['user'] = user  
  df['position'] = position
  df['dataset'] = dataset
  df['filename'] = fname

  #################  Get raw data  ######################

  # Divide raw data and derived features based on time intervals
  x_slices = get_timeslices(timestamp, x, time_interval)
  y_slices = get_timeslices(timestamp, y, time_interval)
  z_slices = get_timeslices(timestamp, z, time_interval)
  ENMO_slices = get_timeslices(timestamp, ENMO, time_interval)
  angz_slices = get_timeslices(timestamp, angle_z, time_interval)
  LIDS_slices = get_timeslices(timestamp, LIDS, time_interval)
 

  # Get raw data slices corresponding to valid labels
  x_valid = x_slices[np.isin(label_agg, sleep_states)]
  y_valid = y_slices[np.isin(label_agg, sleep_states)]
  z_valid = z_slices[np.isin(label_agg, sleep_states)]
  ENMO_valid = ENMO_slices[np.isin(label_agg, sleep_states)]
  angz_valid = angz_slices[np.isin(label_agg, sleep_states)]
  LIDS_valid = LIDS_slices[np.isin(label_agg, sleep_states)]
  
  # Reshape data and labels
  # Data (num_samples x num_timesteps x num_channels)
  num_samples = x_valid.shape[0]; num_tsteps = x_valid.shape[1]
  x_valid = x_valid.reshape((num_samples, num_tsteps, 1))
  y_valid = y_valid.reshape((num_samples, num_tsteps, 1))
  z_valid = z_valid.reshape((num_samples, num_tsteps, 1))
  ENMO_valid = ENMO_valid.reshape((num_samples, num_tsteps, 1))
  angz_valid = angz_valid.reshape((num_samples, num_tsteps, 1))
  LIDS_valid = LIDS_valid.reshape((num_samples, num_tsteps, 1))
  raw_data = np.dstack((x_valid, y_valid, z_valid, ENMO_valid, angz_valid, LIDS_valid))

  # Resample raw data to desired number of timesteps
  raw_data = resample_timeslices(raw_data, num_timesteps)

  return df, raw_data

def main(argv):
  indir = argv[0]
  time_interval = float(argv[1]) # time interval of feature aggregation in seconds
  num_timesteps = int(argv[2]) # number of timesteps in raw data (must not be below 30Hz)
  dataset = argv[3]
  outdir = argv[4]
  
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  
  # Sleep states
  sleep_states = ['Wake','NREM 1','NREM 2','NREM 3','REM','Nonwear']
  
  samp = 0
  files = os.listdir(indir)
  for idx,fname in enumerate(files):
    print('Processing ' + fname)
    
    df, data = process_file(os.path.join(indir, fname), time_interval, sleep_states, dataset, num_timesteps)
    fsamp = data.shape[0]

    # Save features to CSV
    if idx == 0:
      df.to_csv(os.path.join(outdir,'features_' + str(time_interval) + 's.csv'),
                sep=',', mode='w', index=False, header=True)
      with h5py.File(os.path.join(outdir, 'rawdata_'+str(time_interval)+'s.h5'), 'w') as fp:
        fp.create_dataset('data', data=data, compression='gzip', chunks=True,\
                          maxshape=(None,data.shape[1],data.shape[2]))
    else:
      df.to_csv(os.path.join(outdir,'features_' + str(time_interval) + 's.csv'),
                sep=',', mode='a', index=False, header=False)
      with h5py.File(os.path.join(outdir, 'rawdata_'+str(time_interval)+'s.h5'), 'a') as fp:
        fp['data'].resize((fp['data'].shape[0] + data.shape[0]), axis=0)
        fp['data'][-data.shape[0]:] = data
    
if __name__ == "__main__":
  main(sys.argv[1:])
