import sys,os
import numpy as np
import pandas as pd
import h5py, math
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
  return pd.Series(dom_categ['category'])[1:-1]   

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
  slices = np.stack(chan_slices)
  # Use previous and next time intervals along with current intervals
  nsamples, timesteps = slices.shape
  prevnext_slices = np.zeros((nsamples-2, timesteps*3))
  for i in range(1, nsamples-2):
    prevnext_slices[i-1,:timesteps] = slices[i-1] # previous interval
    prevnext_slices[i-1,timesteps:2*timesteps] = slices[i] # current interval
    prevnext_slices[i-1,2*timesteps:] = slices[i+1] # next interval
  return prevnext_slices

def main(argv):
  indir = argv[0]
  time_interval = float(argv[1])
  dataset = argv[2]
  outdir = argv[3]
 
  if not os.path.exists(outdir):
    os.makedirs(outdir)

  sleep_states = ['Wake', 'NREM 1', 'NREM 2', 'NREM 3', 'REM', 'Wake_ext', 'Nonwear']

  lbl_fp = open(os.path.join(outdir,'labels.txt'),'w')
  lbl_fp.write('filename\tlabels\tuser\n')

  files = os.listdir(indir)
  for fname in files:
    print('Processing ' + fname)
        
    fh = h5py.File(os.path.join(indir,fname), 'r')
    x = np.array(fh['X'])
    y = np.array(fh['Y'])
    z = np.array(fh['Z'])
    timestamp = pd.Series(fh['DateTime']).apply(lambda x: x.decode('utf8'))
    timestamp = pd.to_datetime(timestamp, format='%Y-%m-%d %H:%M:%S.%f')

    # Get transformations
    ENMO = get_ENMO(x,y,z)
    angx, angy, angz = get_tilt_angles(x,y,z)
    LIDS = get_LIDS(timestamp, ENMO)
        
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
    label[label == 'Wakefulness'] = 'Wake'
    # Assume unlabeled data as possible wake scenario as long as it is not nonwear
    label[(~np.isin(label,sleep_states)) & (nonwear == False)] = 'Wake_ext'
    # Add nonwear labels 
    label[(~np.isin(label,sleep_states)) & (nonwear == True)] = 'Nonwear'
          
    # Get data slices and dominant labels/nonwear for given time interval
    label_agg = get_dominant_categ(timestamp, label, time_interval)
    x_slices = get_timeslices(timestamp, x, time_interval)
    y_slices = get_timeslices(timestamp, y, time_interval)
    z_slices = get_timeslices(timestamp, z, time_interval)
    ENMO_slices = get_timeslices(timestamp, ENMO, time_interval)
    angz_slices = get_timeslices(timestamp, angz, time_interval)
    LIDS_slices = get_timeslices(timestamp, LIDS, time_interval)
    
    # Get only values corresponding to valid labels
    x_valid = x_slices[label_agg.isin(sleep_states)]
    y_valid = y_slices[label_agg.isin(sleep_states)]
    z_valid = z_slices[label_agg.isin(sleep_states)]
    ENMO_valid = ENMO_slices[label_agg.isin(sleep_states)]
    angz_valid = angz_slices[label_agg.isin(sleep_states)]
    LIDS_valid = LIDS_slices[label_agg.isin(sleep_states)]
    label_valid = label_agg[label_agg.isin(sleep_states)]

    # Reshape data and labels
    # Data (num_samples x num_timesteps x num_channels)
    num_samples = x_valid.shape[0]; num_timesteps = x_valid.shape[1]
    x_valid = x_valid.reshape((num_samples, num_timesteps, 1))
    y_valid = y_valid.reshape((num_samples, num_timesteps, 1))
    z_valid = z_valid.reshape((num_samples, num_timesteps, 1))
    ENMO_valid = ENMO_valid.reshape((num_samples, num_timesteps, 1))
    angz_valid = angz_valid.reshape((num_samples, num_timesteps, 1))
    LIDS_valid = LIDS_valid.reshape((num_samples, num_timesteps, 1))
    data = np.dstack((x_valid, y_valid, z_valid, ENMO_valid, angz_valid, LIDS_valid))
    
    # Save data, labels and other info to file    
    # PSGNewcastle2015 data
    if dataset == 'Newcastle':
        user = fname.split('_')[0]
        position = fname.split('_')[1]
        dataset = 'Newcastle'    
    elif dataset == 'UPenn':
        user = fname.split('.h5')[0][-4:]
        position = 'NaN'
        dataset = 'UPenn'
    elif dataset == 'AMC':
        user = '_'.join(part for part in fname.split('.h5')[0].split('_')[:2])
        position = 'NaN'
        dataset = 'AMC'
        
    out_fname_path = fname.split('.h5')[0]
    for k in range(num_samples):
      out_fname = out_fname_path + '_' + str(k)
      if os.path.exists(os.path.join(outdir,out_fname)):
        continue
      np.save(os.path.join(outdir,out_fname), data[k])
      lbl_fp.write('{}\t{}\t{}\n'.format(out_fname,label_valid[k],user))

  lbl_fp.close()

if __name__ == "__main__":
  main(sys.argv[1:])
