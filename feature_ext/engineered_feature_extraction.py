# -*- coding: utf-8 -*-
import sys,os
import h5py
import numpy as np
import math
import pandas as pd
from scipy.stats import entropy
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

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
   
def compute_entropy(df, bins=20):
    hist, bin_edges = np.histogram(df, bins=bins)
    p = hist/float(hist.sum())
    ent = entropy(p)
    return ent
    
# Aggregate statistics of features over a given time interval
def get_stats(timestamp, feature, time_interval):
    feat_df = pd.DataFrame(data={'timestamp':timestamp, 'feature':feature})
    feat_df.set_index('timestamp', inplace=True)
    feat_mean = feat_df.resample(str(time_interval)+'S').mean()
    feat_std = feat_df.resample(str(time_interval)+'S').std()
    feat_min = feat_df.resample(str(time_interval)+'S').min()
    feat_max = feat_df.resample(str(time_interval)+'S').max()
    feat_mad = feat_df.resample(str(time_interval)+'S').apply(pd.DataFrame.mad)
    feat_ent1 = feat_df.resample(str(time_interval)+'S').apply(compute_entropy, bins=20)
    feat_ent2 = feat_df.resample(str(time_interval)+'S').apply(compute_entropy, bins=200)
    stats = np.vstack((feat_mean['feature'], feat_std['feature'], feat_min['feature'], 
                       feat_max['feature'], feat_mad['feature'], feat_ent1['feature'], feat_ent2['feature'])).T
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
    
def get_dominant_categ(timestamp, categ, time_interval, default='NaN'):
    categ_df = pd.DataFrame(data={'timestamp':timestamp, 'category':categ})
    categ_df.set_index('timestamp', inplace=True)
    dom_categ = categ_df.resample(str(time_interval)+'S').apply(get_categ, default=default)
    return np.array(dom_categ['category'])   

def get_LIDS(timestamp, ENMO):
    df = pd.concat((timestamp, pd.Series(ENMO)), axis=1)
    df.columns = ['timestamp','ENMO']
    df.set_index('timestamp', inplace=True)
    
    df['ENMO_sub'] = np.where(ENMO < 0.02, 0, ENMO-0.02) # assuming ENMO is in g
    ENMO_sub_smooth = df['ENMO_sub'].rolling('600s').sum() # 10-minute rolling sum
    df['LIDS_unfiltered'] = 100.0 / (ENMO_sub_smooth + 1.0)
    LIDS = df['LIDS_unfiltered'].rolling('1800s').mean().values # 30-minute rolling average
    return LIDS

def create_fig_dir(outdir,dirname):
    fig_dir = os.path.join(outdir,'figures',dirname)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    return fig_dir

# Save time series plot
def save_ts_plot(feature, label, nonwear, states, outfile):     
    strip_samp = int(len(feature)/10)
    nsamp = len(feature)
    nstrips = nsamp // strip_samp
    
    plt.figure(num=1, figsize=(2*nstrips,3*nstrips))
    fig, axes = plt.subplots(nstrips,1,sharex=True,num=1)
    min_ylim = min(feature)
    max_ylim = max(feature)
    for strip in range(nstrips):
        y = feature[strip*strip_samp:(strip+1)*strip_samp]
        lbl = label[strip*strip_samp:(strip+1)*strip_samp]
        nw = nonwear[strip*strip_samp:(strip+1)*strip_samp]
        x = range(len(y))
        axes[strip].plot(x,y, color='black')
        axes[strip].set_ylim(min_ylim, max_ylim)
        st1 = axes[strip].fill_between(x, min_ylim, max_ylim, where=(lbl==states[0]), facecolor='green', alpha=0.3)
        st2 = axes[strip].fill_between(x, min_ylim, max_ylim, where=(lbl==states[1]), facecolor='blue', alpha=0.3)
        st3 = axes[strip].fill_between(x, min_ylim, max_ylim, where=(lbl==states[2]), facecolor='yellow', alpha=0.3)
        st4 = axes[strip].fill_between(x, min_ylim, max_ylim, where=(lbl==states[3]), facecolor='magenta', alpha=0.3)
        st5 = axes[strip].fill_between(x, min_ylim, max_ylim, where=(lbl==states[4]), facecolor='cyan', alpha=0.3)
        axes[strip].fill_between(x, min_ylim, max_ylim, where= nw==True, facecolor='red', alpha=0.5)
    fig.legend((st1,st2,st3,st4,st5), states, loc='lower center', ncol=5)
        
    plt.savefig(outfile)
    plt.close('all')
    
def save_agg_plot(df, col_str, order, outfile):
    plt.figure()
    df[col_str] = df[col_str].astype(float)
    sns.boxplot(x="label", y=col_str, data=df, order=order)
    plt.savefig(outfile)
    plt.close('all')
    
def main(argv):
    indir = argv[0]
    time_interval = float(argv[1]) # time interval of feature aggregation in seconds
    outdir = argv[2]
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    ENMO_dir = create_fig_dir(outdir,'ENMO')
    angz_dir = create_fig_dir(outdir,'angle_z')
    LIDS_dir = create_fig_dir(outdir,'LIDS')
    
    ENMO_mean_dir = create_fig_dir(outdir,'ENMO_mean')
    angz_mean_dir = create_fig_dir(outdir,'angz_mean')
    LIDS_mean_dir = create_fig_dir(outdir,'LIDS_mean')
    
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
        ENMO_stats = get_stats(timestamp, ENMO, time_interval)
        angle_z_stats = get_stats(timestamp, angle_z, time_interval)
        LIDS_stats = get_stats(timestamp, LIDS, time_interval)
        feat = np.hstack((ENMO_stats, angle_z_stats, LIDS_stats))
               
        # Get nonwear for each interval
        nonwear = np.array(fh['Nonwear'])
        nonwear_agg = get_dominant_categ(timestamp, nonwear, time_interval, default=True)
        
        # Standardize label names for both datasets
        # Get label for each interval
        label = np.array([x.decode('utf8') for x in np.array(fh['SleepState'])], dtype=object)
        label[label == 'W'] = 'Wake'
        label[label == 'N1'] = 'NREM 1'
        label[label == 'N2'] = 'NREM 2'
        label[label == 'N3'] = 'NREM 3'
        label[label == 'R'] = 'REM'
        label_agg = get_dominant_categ(timestamp, label, time_interval)
          
        # Get valid features and labels
        feat_valid = feat[(nonwear_agg == False) & (label_agg != 'NaN'),:]
        label_valid = label_agg[(nonwear_agg == False) & (label_agg != 'NaN')]
        
        # Write features to CSV file
        data = np.hstack((feat_valid, label_valid.reshape(-1,1)))
        cols = ['ENMO_mean','ENMO_std','ENMO_min','ENMO_max','ENMO_mad','ENMO_entropy1','ENMO_entropy1', 
                'angz_mean','angz_std','angz_min','angz_max','angz_mad','angz_entropy1','angz_entropy2', 
                'LIDS_mean','LIDS_std','LIDS_min','LIDS_max','LIDS_mad','LIDS_entropy1','LIDS_entropy2','label']
        df = pd.DataFrame(data=data, columns=cols)
        
#        # Uncomment for PSGNewcastle2015 data
#        user = fname.split('_')[0]
#        position = fname.split('_')[1]
#        dataset = 'Newcastle'        
        # Uncomment for UPenn_Axivity data
        user = fname.split('.h5')[0][-4:]
        position = 'NaN'
        dataset = 'UPenn'
        
        df['user'] = user  
        df['position'] = position
        df['dataset'] = dataset
        
        # Save data to CSV
        if idx == 0:
            df.to_csv(os.path.join(outdir,'sleep_data_' + str(time_interval) + 's.csv'), sep=',', mode='w', index=False, header=True)
        else:
            df.to_csv(os.path.join(outdir,'sleep_data_' + str(time_interval) + 's.csv'), sep=',', mode='a', index=False, header=False)
        
        ############## Plot features #####################

        # Plot features after aggregation
        save_ts_plot(ENMO_stats[:,0], label_agg, nonwear_agg, states, os.path.join(ENMO_dir,fname.split('.h5')[0]+'.jpg'))
        save_ts_plot(angle_z_stats[:,0], label_agg, nonwear_agg, states, os.path.join(angz_dir,fname.split('.h5')[0]+'.jpg')) 
        save_ts_plot(LIDS_stats[:,0], label_agg, nonwear_agg, states, os.path.join(LIDS_dir,fname.split('.h5')[0]+'.jpg')) 
        
        save_agg_plot(df, 'ENMO_mean', states, os.path.join(ENMO_mean_dir,fname.split('.h5')[0]+'.jpg'))
        save_agg_plot(df, 'angz_mean', states, os.path.join(angz_mean_dir,fname.split('.h5')[0]+'.jpg'))
        save_agg_plot(df, 'LIDS_mean', states, os.path.join(LIDS_mean_dir,fname.split('.h5')[0]+'.jpg'))
        
        #break    
    
if __name__ == "__main__":
    main(sys.argv[1:])
