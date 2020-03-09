# -*- coding: utf-8 -*-
import sys,os
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
