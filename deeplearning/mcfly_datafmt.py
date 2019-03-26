import sys,os
import numpy as np
import pandas as pd
import h5py
from collections import Counter

def get_categ(df, default='NaN'):
  if len(df) == 0:
    print('zero length')
    return default
  ctr = Counter(df)
  for key in ctr:
    ctr[key] = ctr[key]/float(len(df))
  dom_categ = ctr.most_common()[0]
  if dom_categ[1] >= 0.7: # If a category occurs more than 70% of time interval, mark that as dominant category
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

def get_timeslices(timestamp, channel, time_interval):
  chan_df = pd.DataFrame(data={'timestamp':timestamp, 'channel':channel})
  chan_df.set_index('timestamp', inplace=True)
  chan_slices = chan_df.resample(str(time_interval)+'S').apply(get_tslice)
  lengths = [tslice.shape[0] for tslice in chan_slices]
  resize_len = int(np.median(lengths))
  for tslice in chan_slices:
    tslice.resize((resize_len), refcheck=False) 
  return np.stack(chan_slices)

def main(argv):
  indir = argv[0]
  time_interval = float(argv[1])
  outdir = argv[2]
 
  if not os.path.exists(outdir):
    os.makedirs(outdir)

  sleep_states = ['Wake', 'NREM 1', 'NREM 2', 'NREM 3', 'REM']

  files = os.listdir(indir)
  for fname in files:
    print('Processing ' + fname)
        
    fh = h5py.File(os.path.join(indir,fname), 'r')
    x = np.array(fh['X'])
    y = np.array(fh['Y'])
    z = np.array(fh['Z'])
    timestamp = pd.Series(fh['DateTime']).apply(lambda x: x.decode('utf8'))
    timestamp = pd.to_datetime(timestamp, format='%Y-%m-%d %H:%M:%S.%f')
        
    # Get nonwear for each interval
    nonwear = np.array(fh['Nonwear'])
        
    # Standardize label names for both datasets
    # Get label for each interval
    label = np.array([x.decode('utf8') for x in np.array(fh['SleepState'])], dtype=object)
    label[label == 'W'] = 'Wake'
    label[label == 'N1'] = 'NREM 1'
    label[label == 'N2'] = 'NREM 2'
    label[label == 'N3'] = 'NREM 3'
    label[label == 'R'] = 'REM'
          
    # Get data slices and dominant labels/nonwear for given time interval
    label_agg = get_dominant_categ(timestamp, label, time_interval)
    nonwear_agg = get_dominant_categ(timestamp, nonwear, time_interval)
    x_slices = get_timeslices(timestamp, x, time_interval)
    y_slices = get_timeslices(timestamp, y, time_interval)
    z_slices = get_timeslices(timestamp, z, time_interval)

    # Get only values corresponding to not nonwear and valid labels
    x_valid = x_slices[(nonwear_agg == False) & (label_agg.isin(sleep_states))]
    y_valid = y_slices[(nonwear_agg == False) & (label_agg.isin(sleep_states))]
    z_valid = z_slices[(nonwear_agg == False) & (label_agg.isin(sleep_states))]
    label_valid = label_agg[(nonwear_agg == False) & (label_agg.isin(sleep_states))]

    # Reshape data and labels
    # Data (num_samples x num_timesteps x num_channels)
    num_samples = x_valid.shape[0]; num_timesteps = x_valid.shape[1]
    x_valid = x_valid.reshape((num_samples, num_timesteps, 1))
    y_valid = y_valid.reshape((num_samples, num_timesteps, 1))
    z_valid = z_valid.reshape((num_samples, num_timesteps, 1))
    data = np.dstack((x_valid, y_valid, z_valid))
    # Labels (num_samples x num_classes)
    labels = np.zeros((num_samples, len(sleep_states)))
    for idx,lbl in enumerate(label_valid):
      labels[idx,sleep_states.index(lbl)] = 1
    
    # Save data, labels and other info to file    
    # Uncomment for PSGNewcastle2015 data
    user = fname.split('_')[0]
    position = fname.split('_')[1]
    dataset = 'Newcastle'        
    # Uncomment for UPenn_Axivity data
#    user = fname.split('.h5')[0][-4:]
#    position = 'NaN'
#    dataset = 'UPenn'
        
    out_fname = fname.split('.h5')[0] + '.npz'
    np.savez(os.path.join(outdir,out_fname), data=data, labels=labels, user=user, position=position, dataset=dataset)
    print(data.shape)

if __name__ == "__main__":
  main(sys.argv[1:])
