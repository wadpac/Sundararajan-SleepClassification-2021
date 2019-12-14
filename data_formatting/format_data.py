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

def compute_entropy(df, bins=20):
  hist, bin_edges = np.histogram(df, bins=bins)
  p = hist/float(hist.sum())
  ent = entropy(p)
  return ent

# Get difference of feature with respect to prev or next interval
def get_diff_feat(feature, direction='prev'):
  diff = np.zeros(len(feature))
  if direction == 'prev':
    for i in range(1,len(feature)):
      diff[i] = feature[i]-feature[i-1]
  else:
    for i in range(len(feature)-1):
      diff[i] = feature[i+1]-feature[i]

  return diff
    
# Aggregate statistics of features over a given time interval
def get_stats(timestamp, feature, time_interval):
  feat_df = pd.DataFrame(data={'timestamp':timestamp, 'feature':feature})
  feat_df.set_index('timestamp', inplace=True)
  feat_mean = feat_df.resample(str(time_interval)+'S').mean()
  feat_std = feat_df.resample(str(time_interval)+'S').std()
  feat_min = feat_df.resample(str(time_interval)+'S').min()
  feat_max = feat_df.resample(str(time_interval)+'S').max()
  feat_mad = feat_df.resample(str(time_interval)+'S').apply(pd.DataFrame.mad)
  feat_ent1 = feat_df.resample(str(time_interval)+'S')\
                              .apply(compute_entropy, bins=20)
  feat_ent2 = feat_df.resample(str(time_interval)+'S')\
                              .apply(compute_entropy, bins=200)
  feat_prevdiff = get_diff_feat(feat_mean['feature'], 'prev')
  feat_nextdiff = get_diff_feat(feat_mean['feature'], 'next')
  stats = np.vstack((feat_mean['feature'], feat_std['feature'], feat_min['feature'], 
                     feat_max['feature'], feat_mad['feature'], feat_ent1['feature'], 
                     feat_ent2['feature'], feat_prevdiff, feat_nextdiff)).T

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
  dom_categ = categ_df.resample(str(time_interval)+'S').apply(get_categ, default=default)
  return pd.Series(dom_categ['category'])   

def get_tslice(df):
  tslice = np.array(df['channel'])
  return tslice

def get_rawdata_shape(indir, time_interval, sleep_states):
  num_samples = 0 
  files = os.listdir(indir)
  for idx,fname in enumerate(files):
    print('Touching ' + fname)
    fh = h5py.File(os.path.join(indir,fname), 'r')
    
    # Get raw data from file - x,timestamp        
    x = np.array(fh['X'])
    timestamp = pd.Series(fh['DateTime']).apply(lambda x: x.decode('utf8'))
    timestamp = pd.to_datetime(timestamp, format='%Y-%m-%d %H:%M:%S.%f')
    
    # Get nonwear for each interval
    nonwear = np.array(fh['Nonwear'])
    
    # Standardize label names for all datasets
    # Get label for each interval
    label = np.array([x.decode('utf8') for x in np.array(fh['SleepState'])],
                     dtype=object)
    label[label == 'W'] = 'Wake'
    label[label == 'N1'] = 'NREM 1'
    label[label == 'N2'] = 'NREM 2'
    label[label == 'N3'] = 'NREM 3'
    label[label == 'R'] = 'REM'
    label[label == 'Wakefulness'] = 'Wake'
      
    # Assume unlabeled data as possible wake scenario as long as it is not nonwear
    label[(~np.isin(label,sleep_states)) & (nonwear == False)] = 'Wake_ext'
    # Add nonwear labels 
    label[(~np.isin(label,sleep_states)) & (nonwear == True)] = 'Nonwear'
   
    label_agg = get_dominant_categ(timestamp, label, time_interval)
    x_slices = get_timeslices(timestamp, x, time_interval)
    x_valid = x_slices[label_agg.isin(sleep_states)]
    num_samples += x_valid.shape[0]
 
  return num_samples

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

  # Get raw data from file - x,y,z,timestamp        
  x = np.array(fh['X'])
  y = np.array(fh['Y'])
  z = np.array(fh['Z'])
  timestamp = pd.Series(fh['DateTime']).apply(lambda x: x.decode('utf8'))
  timestamp = pd.to_datetime(timestamp, format='%Y-%m-%d %H:%M:%S.%f')
  
  # Get derived features - ENMO, angle_z, LIDS
  ENMO = get_ENMO(x,y,z)
  angle_x, angle_y, angle_z = get_tilt_angles(x,y,z)
  LIDS = get_LIDS(timestamp, ENMO)
  
  # Get nonwear for each interval
  nonwear = np.array(fh['Nonwear'])
  nonwear_agg = get_dominant_categ(timestamp, nonwear, time_interval)
  
  # Standardize label names for all datasets
  # Get label for each interval
  label = np.array([x.decode('utf8') for x in np.array(fh['SleepState'])],
                   dtype=object)
  label[label == 'W'] = 'Wake'
  label[label == 'N1'] = 'NREM 1'
  label[label == 'N2'] = 'NREM 2'
  label[label == 'N3'] = 'NREM 3'
  label[label == 'R'] = 'REM'
  label[label == 'Wakefulness'] = 'Wake'
    
  # Assume unlabeled data as possible wake scenario as long as it is not nonwear
  label[(~np.isin(label,sleep_states)) & (nonwear == False)] = 'Wake_ext'
  # Add nonwear labels 
  label[(~np.isin(label,sleep_states)) & (nonwear == True)] = 'Nonwear'
 
  # Aggregate data over time intervals and choose only intervals corresponding
  # to valid labels
  label_agg = get_dominant_categ(timestamp, label, time_interval)
    
  # Get statistics of features for given time intervals
  _, ENMO_stats = get_stats(timestamp, ENMO, time_interval)
  _, angle_z_stats = get_stats(timestamp, angle_z, time_interval)
  timestamp_agg, LIDS_stats = get_stats(timestamp, LIDS, time_interval)
  feat = np.hstack((ENMO_stats, angle_z_stats, LIDS_stats))
  
  # Get valid timestamps, features
  timestamp_valid = timestamp_agg[label_agg.isin(sleep_states)].reshape(-1,1)
  feat_valid = feat[label_agg.isin(sleep_states),:]
  
  # Divide raw data and derived features based on time intervals
  x_slices = get_timeslices(timestamp, x, time_interval)
  y_slices = get_timeslices(timestamp, y, time_interval)
  z_slices = get_timeslices(timestamp, z, time_interval)
  ENMO_slices = get_timeslices(timestamp, ENMO, time_interval)
  angz_slices = get_timeslices(timestamp, angle_z, time_interval)
  LIDS_slices = get_timeslices(timestamp, LIDS, time_interval)
 

  # Get raw data slices corresponding to valid labels
  x_valid = x_slices[label_agg.isin(sleep_states)]
  y_valid = y_slices[label_agg.isin(sleep_states)]
  z_valid = z_slices[label_agg.isin(sleep_states)]
  ENMO_valid = ENMO_slices[label_agg.isin(sleep_states)]
  angz_valid = angz_slices[label_agg.isin(sleep_states)]
  LIDS_valid = LIDS_slices[label_agg.isin(sleep_states)]
  
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

  nonwear_valid = nonwear_agg[label_agg.isin(sleep_states)].values
  label_valid = label_agg[label_agg.isin(sleep_states)].values
  
  # Write features to CSV file
  data = np.hstack((timestamp_valid.reshape(-1,1), feat_valid, label_valid.reshape(-1,1), nonwear_valid.reshape(-1,1)))
  cols = ['timestamp','ENMO_mean','ENMO_std','ENMO_min','ENMO_max','ENMO_mad',
          'ENMO_entropy1','ENMO_entropy2','ENMO_prevdiff','ENMO_nextdiff', 
          'angz_mean','angz_std','angz_min','angz_max','angz_mad',
          'angz_entropy1','angz_entropy2','angz_prevdiff','angz_nextdiff', 
          'LIDS_mean','LIDS_std','LIDS_min','LIDS_max','LIDS_mad',
          'LIDS_entropy1','LIDS_entropy2','LIDS_prevdiff','LIDS_nextdiff','label','nonwear']
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
  df['filename'] = filename

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
  sleep_states = ['Wake','NREM 1','NREM 2','NREM 3','REM','Nonwear','Wake_ext']
 
  num_samples = get_rawdata_shape(indir, time_interval, sleep_states)
  num_channels = 6
  raw_data = np.memmap(os.path.join(outdir, 'rawdata_' + str(time_interval) + 's.npz'), dtype='float32',
                       mode='w+', shape=(num_samples, num_timesteps, num_channels))
  with open(os.path.join(outdir,'datashape_'+ str(time_interval) + 's.csv'), 'w') as fp: 
    fp.write('num_samples,num_timesteps,num_channels\n')
    fp.write('%d,%d,%d\n' % (num_samples,num_timesteps,num_channels))

  files = os.listdir(indir)
  samp = 0
  for idx,fname in enumerate(files):
    print('Processing ' + fname)
 
    df, data = process_file(os.path.join(indir, fname), time_interval, sleep_states, dataset, num_timesteps)
    fsamp = data.shape[0]

    # Save features to CSV
    if idx == 0:
      df.to_csv(os.path.join(outdir,'features_' + str(time_interval) + 's.csv'),
                sep=',', mode='w', index=False, header=True)
    else:
      df.to_csv(os.path.join(outdir,'features_' + str(time_interval) + 's.csv'),
                sep=',', mode='a', index=False, header=False)

    # Save raw data to numpy memmap file
    raw_data[samp:samp+fsamp,:,:] = data[:,:,:]
    samp += fsamp
  print(num_samples, samp)  

  # Flush memmap file to write contents to disk
  del raw_data
    
if __name__ == "__main__":
  main(sys.argv[1:])
