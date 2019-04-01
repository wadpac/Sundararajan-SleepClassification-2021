import sys,os
import numpy as np
import pandas as pd
import h5py

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

def main(argv):
  indir = argv[0]
  seq_interval = int(argv[1]) # Time interval in seconds for a sequence 
  token_interval = int(argv[2]) # Time interval in seconds for tokens in a seq
  outdir = argv[3]

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
        
        
    

if __name__ == '__main__':
  main(sys.argv[1:])
