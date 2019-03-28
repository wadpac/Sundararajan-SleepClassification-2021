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
    feat_ent1 = feat_df.resample(str(time_interval)+'S').apply(compute_entropy, bins=20)
    feat_ent2 = feat_df.resample(str(time_interval)+'S').apply(compute_entropy, bins=200)
    feat_prevdiff = get_diff_feat(feat_mean['feature'], 'prev')
    feat_nextdiff = get_diff_feat(feat_mean['feature'], 'next')
    stats = np.vstack((feat_mean['feature'], feat_std['feature'], feat_min['feature'], 
                       feat_max['feature'], feat_mad['feature'], feat_ent1['feature'], 
                       feat_ent2['feature'], feat_prevdiff, feat_nextdiff)).T
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

def main(argv):
    indir = argv[0]
    time_interval = float(argv[1]) # time interval of feature aggregation in seconds
    outdir = argv[2]
    
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
        cols = ['ENMO_mean','ENMO_std','ENMO_min','ENMO_max','ENMO_mad','ENMO_entropy1','ENMO_entropy2','ENMO_prevdiff','ENMO_nextdiff', 
                'angz_mean','angz_std','angz_min','angz_max','angz_mad','angz_entropy1','angz_entropy2','angz_prevdiff','angz_nextdiff', 
                'LIDS_mean','LIDS_std','LIDS_min','LIDS_max','LIDS_mad','LIDS_entropy1','LIDS_entropy2','LIDS_prevdiff','LIDS_nextdiff','label']
        df = pd.DataFrame(data=data, columns=cols)
        
        # Uncomment for PSGNewcastle2015 data
        user = fname.split('_')[0]
        position = fname.split('_')[1]
        dataset = 'Newcastle'        
        # Uncomment for UPenn_Axivity data
#        user = fname.split('.h5')[0][-4:]
#        position = 'NaN'
#        dataset = 'UPenn'
        
        df['user'] = user  
        df['position'] = position
        df['dataset'] = dataset
        
        # Save data to CSV
        if idx == 0:
            df.to_csv(os.path.join(outdir,'sleep_data_' + str(time_interval) + 's.csv'), sep=',', mode='w', index=False, header=True)
        else:
            df.to_csv(os.path.join(outdir,'sleep_data_' + str(time_interval) + 's.csv'), sep=',', mode='a', index=False, header=False)
        
        #break    
    
if __name__ == "__main__":
    main(sys.argv[1:])
