import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import math
import numpy as np
import pandas as pd

from utils import get_ENMO, get_tilt_angles, get_LIDS
from utils import mad, compute_entropy, get_diff_feat, get_stats
  
def compute_features(data, time_interval):
  df = pd.DataFrame(data, columns=['timestamp','x','y','z'])
  df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
  
  x = np.array(df['x'])
  y = np.array(df['y'])
  z = np.array(df['z'])
  timestamp = pd.Series(df['timestamp'])
  
  # Perform flipping x and y axes to ensure standard orientation
  # For correct orientation, x-angle should be mostly negative
  # So, if median x-angle is positive, flip both x and y axes
  # Ref: https://github.com/wadpac/hsmm4acc/blob/524743744068e83f468a4e217dde745048a625fd/UKMovementSensing/prepacc.py
  angx = np.arctan2(x, np.sqrt(y*y + z*z)) * 180.0/math.pi
  if np.median(angx) > 0:
      x *= -1
      y *= -1

  ENMO = get_ENMO(x,y,z)
  angle_x, angle_y, angle_z = get_tilt_angles(x,y,z)
  LIDS = get_LIDS(timestamp, ENMO)
  
  _, ENMO_stats = get_stats(df['timestamp'], ENMO, time_interval)
  _, angle_z_stats = get_stats(df['timestamp'], angle_z, time_interval)
  timestamp_agg, LIDS_stats = get_stats(df['timestamp'], LIDS, time_interval)
  feat = np.hstack((ENMO_stats, angle_z_stats, LIDS_stats))

  return feat
