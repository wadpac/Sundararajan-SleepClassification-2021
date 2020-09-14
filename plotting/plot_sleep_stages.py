import sys,os
import h5py
import numpy as np
import pandas as pd
import itertools, operator
import matplotlib.pyplot as plt

def plot_intervals(ax, bin_data, facecolor='white', alpha=0.5, label=None):
  handle = None
  intervals = [[i for i,val in it] for key,it in itertools.groupby(enumerate(bin_data),\
                key=operator.itemgetter(1)) if key != 0]
  for interval in intervals:
    handle = ax.axvspan(interval[0], interval[-1], facecolor=facecolor, alpha=alpha, label=label)
  return handle

def estimate_nonwear(timestamp, x, y, z, interval='900', th=0.013):
  df = pd.DataFrame(data={'timestamp':timestamp, 'x':x, 'y':y, 'z':z})
  df.set_index('timestamp', inplace=True)
  df_std = df.resample(str(interval)+'S').std()
  is_nonwear = [(sum([row.x < th, row.y < th, row.z <th]) >=2) for index, row in df_std.iterrows()]
  df_std['nonwear'] = is_nonwear
  
  nonwear_df = pd.DataFrame(data={'timestamp':timestamp})
  nonwear_df['nonwear'] = False
  intervals = list(df_std.index.values)
  df_std = df_std.reset_index()
  for index,ts in enumerate(intervals):
    if index < len(intervals)-1:
      nonwear_df.loc[(nonwear_df['timestamp'] >= intervals[index]) &\
                     (nonwear_df['timestamp'] < intervals[index+1]),\
                     'nonwear'] = df_std.iloc[index]['nonwear']
  nonwear_df.loc[nonwear_df['timestamp'] >= intervals[-1], 'nonwear'] = df_std.iloc[-1]['nonwear']
  return np.array(nonwear_df['nonwear'].values)

def main(argv):
  indir = argv[0] # input directory containing preprocessed hdf5 files
  outdir = argv[1] # output directory to store plots

  if not os.path.exists(outdir):
    os.makedirs(outdir)
 
  sleep_stages = ['Wake','NREM1','NREM2','NREM3','REM','Nonwear','Unlabeled']
  valid_states = ['Wake','NREM 1','NREM 2','NREM 3','REM']

  files = os.listdir(indir)
  for fname in files:
    out_fname = fname.split('.h5')[0] + '.jpg'
    #if os.path.exists(os.path.join(outdir, out_fname)):
    #  continue

    fh = h5py.File(os.path.join(indir,fname), 'r')
    x = np.array(fh['X'])
    y = np.array(fh['Y'])
    z = np.array(fh['Z'])
    
    # Normalize accelerometer data
    #x = (x-x.mean())/(x.std())
    #y = (y-y.mean())/(y.std())
    #z = (z-z.mean())/(z.std())
  
    timestamp = pd.Series(fh['DateTime']).apply(lambda x: x.decode('utf8'))
    timestamp = pd.to_datetime(timestamp, format='%Y-%m-%d %H:%M:%S.%f')
    #nonwear = np.array(fh['Nonwear'])
    nonwear = estimate_nonwear(timestamp, x, y, z)
    
    label = np.array([x.decode('utf8') for x in np.array(fh['SleepState'])], dtype=object)
    label[label == 'W'] = 'Wake'
    label[label == 'N1'] = 'NREM 1'
    label[label == 'N2'] = 'NREM 2'
    label[label == 'N3'] = 'NREM 3'
    label[label == 'R'] = 'REM'
    label[label == 'Wakefulness'] = 'Wake'
    
    # Remove labeled intervals from nonwear
    label[(nonwear == True) & (~np.isin(label, valid_states))] = 'Nonwear'

    nsamples = x.shape[0]
    
    # Plot entire data as subplots if long
    num_subplots = 5
    span = nsamples // num_subplots
    if nsamples % span:
        num_subplots += 1
    print('{}: #samples = {}'.format(fname, nsamples))
      
    fig,axes = plt.subplots(num_subplots, 1, figsize=(70,10*num_subplots), sharey=True)
    legend_info = []
    for i in range(num_subplots):
      span_len = x[i*span:(i+1)*span].shape[0] # will be less than span for last subplot
      axes[i].plot(np.arange(span_len), x[i*span:(i+1)*span], 'r-', linewidth=5)
      axes[i].plot(np.arange(span_len), y[i*span:(i+1)*span], 'g-', linewidth=5)
      axes[i].plot(np.arange(span_len), z[i*span:(i+1)*span], 'b-', linewidth=5)

      sub_timestamp = timestamp[i*span:(i+1)*span]
      sub_timestamp = [ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] for ts in sub_timestamp] 
      nticks = len(axes[i].get_xticklabels())-2 # plot doesnt seem to display first and last tick
      step = span//(nticks-1)
      sub_ts = [0] + list(sub_timestamp[::step]) + [sub_timestamp[-1]] + [0]
      axes[i].set_xticklabels(sub_ts, fontsize=30)
      axes[i].yaxis.set_tick_params(labelsize=40)
      
      sub_nonwear = label[i*span:(i+1)*span] == 'Nonwear'
      handle = plot_intervals(axes[i], sub_nonwear, 'red', label='Nonwear')
      if handle is not None:
        legend_info.append((handle,'Nonwear'))
     
      sub_wake = label[i*span:(i+1)*span] == 'Wake'
      handle = plot_intervals(axes[i], sub_wake, 'green', label='Wake')
      if handle is not None:
        legend_info.append((handle,'Wake'))
      
      sub_rem = label[i*span:(i+1)*span] == 'REM'
      handle = plot_intervals(axes[i], sub_rem, 'blue', label='REM')
      if handle is not None:
        legend_info.append((handle,'REM'))
  
      sub_nrem1 = label[i*span:(i+1)*span] == 'NREM 1'
      handle = plot_intervals(axes[i], sub_nrem1, 'cyan', label='NREM 1')
      if handle is not None:
        legend_info.append((handle,'NREM1'))
  
      sub_nrem2 = label[i*span:(i+1)*span] == 'NREM 2'
      handle = plot_intervals(axes[i], sub_nrem2, 'yellow', label='NREM 2')
      if handle is not None:
        legend_info.append((handle,'NREM2'))
  
      sub_nrem3 = label[i*span:(i+1)*span] == 'NREM 3'
      handle = plot_intervals(axes[i], sub_nrem3, 'magenta', label='NREM 3')
      if handle is not None:
        legend_info.append((handle,'NREM3'))
  
      sub_unlbl = label[i*span:(i+1)*span] == 'NaN'
      handle = plot_intervals(axes[i], sub_unlbl, 'black', alpha=0.2, label='Unlabeled')
      if handle is not None:
        legend_info.append((handle,'Unlabeled'))
  
    handles, legend_lbl = list(zip(*legend_info))
    unique_legend_info = [(handles[legend_lbl.index(lbl)],lbl) for lbl in sleep_stages \
                            if lbl in set(legend_lbl)]
    handles, legend_lbl = list(zip(*unique_legend_info))
    plt.figlegend(handles, legend_lbl, loc='lower center', ncol=len(legend_lbl), fontsize=70)
    plt.savefig(os.path.join(outdir,out_fname), bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
  main(sys.argv[1:])
