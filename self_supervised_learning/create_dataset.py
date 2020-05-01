import sys,os
import numpy as np
import pandas as pd
import math
import h5py
import argparse
import random
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

def rand_sample_timesteps(X, steps=1000):
  tt = np.zeros((steps,), dtype=int)
  tt[:] = np.sort(np.random.randint(1,X.shape[0]-1,steps),axis=0)
  tt[:] = X.shape[0]-1
  return tt

def rand_sampling(X, steps=1000):
  tt = rand_sample_timesteps(X, steps)
  X_new = np.zeros((steps,X.shape[1]))
  for i in range(X.shape[1]):
    X_new[:,i] = np.interp(tt, np.arange(X.shape[0]), X[:,i])
  return X_new

def get_tslice(df):
  tslice = np.array(df[df.columns])
  return tslice

def get_pairs(df, span=30, steps=1500, tpos=180, tneg=360):
  # Get time slices according to span and remove those intervals with very few steps
  slices = df.resample(str(span)+'S').apply(get_tslice)
  remove_idx = []
  for idx,item in slices.iteritems():
    if np.array(item).shape[0] < int(0.75*steps):
      remove_idx.append(idx)
  slices = slices.drop(labels=remove_idx)

  channels = len(df.columns)
  samp1 = np.zeros((2*len(slices), steps, channels))
  samp2 = np.zeros((2*len(slices), steps, channels))
  lbl = np.zeros((2*len(slices),))
  samp = 0
  for idx,item in slices.iteritems():
    # Pick a positive sample  
    start_time = idx - pd.Timedelta(value=tpos, unit='S')
    end_time = idx + pd.Timedelta(value=tpos, unit='S')
    pos_indices = slices.index[(slices.index >= start_time) & (slices.index <= end_time) & (slices.index != idx)]
    if len(pos_indices):
      pos_idx = random.choice(pos_indices)
      samp1[samp] = rand_sampling(slices[slices.index == idx].values[0], steps)
      samp2[samp] = rand_sampling(slices[slices.index == pos_idx].values[0], steps)
      lbl[samp] = 1
      samp += 1

    # Pick a negative sample  
    start_time1 = idx - pd.Timedelta(value=2*tneg, unit='S')
    end_time1 = idx - pd.Timedelta(value=tneg, unit='S')
    start_time2 = idx + pd.Timedelta(value=tneg, unit='S')
    end_time2 = idx + pd.Timedelta(value=2*tneg, unit='S')
    neg_indices = slices.index[(((slices.index >= start_time1) & (slices.index <= end_time1)) |\
                               ((slices.index >= start_time2) & (slices.index <= end_time2))) &\
                               (slices.index != idx)]
    if len(neg_indices):
      neg_idx = random.choice(neg_indices)
      samp1[samp] = rand_sampling(slices[slices.index == neg_idx].values[0], steps)
      samp2[samp] = rand_sampling(slices[slices.index == idx].values[0], steps)
      lbl[samp] = 0
      samp += 1
  
  # Resize samples
  samp1 = samp1[:samp]
  samp2 = samp2[:samp]
  lbl = lbl[:samp]

  # Shuffle data
  indices = np.arange(samp)
  random.shuffle(indices)
  samp1 = samp1[indices]
  samp2 = samp2[indices]
  lbl = lbl[indices]

  return samp1, samp2, lbl

def main(args):
  if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

  # Get sample pairs
  files = os.listdir(args.indir)
  for idx,fname in enumerate(files):
    print('Processing ' + fname)

    fh = h5py.File(os.path.join(args.indir,fname), 'r')
    x = np.array(fh['X'])
    y = np.array(fh['Y'])
    z = np.array(fh['Z'])
    timestamp = pd.Series(fh['DateTime']).apply(lambda x: x.decode('utf8'))
    timestamp = pd.to_datetime(timestamp, format='%Y-%m-%d %H:%M:%S.%f')
  
    if args.channels == 3:
      df = pd.DataFrame({'timestamp':timestamp, 'x':x, 'y':y, 'z':z}) 
    else:
      ENMO = get_ENMO(x,y,z)
      angx, angy, angz = get_tilt_angles(x,y,z)
      LIDS = get_LIDS(timestamp, ENMO)
      df = pd.DataFrame({'timestamp':timestamp, 'x':x, 'y':y, 'z':z,\
                         'ENMO':ENMO, 'angz':angz, 'LIDS':LIDS}) 
    df.set_index('timestamp', inplace=True)
    samp1, samp2, lbl = get_pairs(df, args.span, args.steps, args.tpos, args.tneg)
    if idx == 0:
      with h5py.File(os.path.join(args.outdir, 'dataset.h5'), 'w') as fp:
        fp.create_dataset('samp1', data=samp1, compression='gzip', chunks=True,\
                          maxshape=(None,samp1.shape[1],samp1.shape[2]))
        fp.create_dataset('samp2', data=samp2, compression='gzip', chunks=True,\
                          maxshape=(None,samp2.shape[1],samp2.shape[2]))
        fp.create_dataset('label', data=lbl, compression='gzip', chunks=True,\
                          maxshape=(None,))
    else:
      with h5py.File(os.path.join(args.outdir, 'dataset.h5'), 'a') as fp:
        fp['samp1'].resize((fp['samp1'].shape[0] + samp1.shape[0]), axis=0)
        fp['samp1'][-samp1.shape[0]:] = samp1
        fp['samp2'].resize((fp['samp2'].shape[0] + samp2.shape[0]), axis=0)
        fp['samp2'][-samp2.shape[0]:] = samp2
        fp['label'].resize((fp['label'].shape[0] + lbl.shape[0]), axis=0)
        fp['label'][-lbl.shape[0]:] = lbl

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--indir', type=str, help='Input directory')
  parser.add_argument('--span', type=int, default=30, help='Sample span in seconds')
  parser.add_argument('--steps', type=int, default=1500, help='No. of timesteps in a sample')
  parser.add_argument('--channels', type=int, default=6, help='No. of channels (3 or 6)')
  parser.add_argument('--tpos', type=int, default=180, help='Window in seconds for a positive sample')
  parser.add_argument('--tneg', type=int, default=360, help='Window in seconds for a negative sample')
  parser.add_argument('--outdir', type=str, help='Output directory')
  args = parser.parse_args()
  main(args)
