# -*- coding: utf-8 -*-
import sys,os
import h5py
import numpy as np
import math
import pandas as pd
from scipy.stats import entropy
from collections import Counter
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters

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
    ENMO_sub_smooth = df['ENMO_sub'].rolling('600s').sum() # 10-minute rolling sum
    df['LIDS_unfiltered'] = 100.0 / (ENMO_sub_smooth + 1.0)
    LIDS = df['LIDS_unfiltered'].rolling('1800s').mean().values # 30-minute rolling average
    return LIDS

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

def get_tsfresh_feat(df, colName=None):
    df = df.reset_index()
    df.columns = ['timestamp',colName]
    df['id'] = 0 # Mandatory for tsfresh to group
    ext_feat = extract_features(df, column_id='id', column_value=colName, column_sort='timestamp', default_fc_parameters=EfficientFCParameters(), disable_progressbar=True)
    ext_feat_val = ext_feat.values[0]
    #print(ext_feat_val, ext_feat.columns); exit()
    return ext_feat_val
   
def get_aggregated_feat(timestamp, time_interval, feat, featName):
    df = pd.DataFrame(list(zip(timestamp,feat)), columns=['timestamp',featName])
    df.set_index('timestamp', inplace=True)
    df_resamp = df.resample(str(time_interval)+'S').apply(get_tsfresh_feat, colName=featName)
    print(list(df_resamp.columns)); exit()
    return 0
 
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

        valid_timestamp = timestamp[(nonwear == False) & (label != 'NaN')]
        valid_ENMO = ENMO[(nonwear == False) & (label != 'NaN')]
        valid_angle_z = angle_z[(nonwear == False) & (label != 'NaN')]
        valid_LIDS = LIDS[(nonwear == False) & (label != 'NaN')]
        valid_label = label[(nonwear == False) & (label != 'NaN')]
        
        ENMO_feat = get_aggregated_feat(timestamp, time_interval, ENMO, 'ENMO')
        angz_feat = get_aggregated_feat(timestamp, time_interval, angle_z, 'angz')
        LIDS_feat = get_aggregated_feat(timestamp, time_interval, LIDS, 'LIDS')
 
        break   

 
if __name__ == "__main__":
    main(sys.argv[1:])
