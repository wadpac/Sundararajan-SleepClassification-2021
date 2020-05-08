import sys,os
import h5py
import numpy as np
import pandas as pd

def main(argv):
  indir = argv[0]
  outdir = argv[1]

  states = ['Wake','NREM 1','NREM 2','NREM 3','REM']
  
  files = os.listdir(indir)
  for fname in files:
    print('Processing ' + fname)
    
    fh = h5py.File(os.path.join(indir,fname), 'r')
    x = np.array(fh['X'])
    y = np.array(fh['Y'])
    z = np.array(fh['Z'])
    timestamp = pd.Series(fh['DateTime'])
   
    label = np.array([x.decode('utf8') for x in np.array(fh['SleepState'])],
                     dtype=object)
    label[label == 'W'] = 'Wake'
    label[label == 'N1'] = 'NREM 1'
    label[label == 'N2'] = 'NREM 2'
    label[label == 'N3'] = 'NREM 3'
    label[label == 'R'] = 'REM'
    label[label == 'Wakefulness'] = 'Wake'

    x = x[~np.isin(label, states)]
    y = y[~np.isin(label, states)]
    z = z[~np.isin(label, states)]
    timestamp = timestamp[~np.isin(label, states)]

    with h5py.File(os.path.join(outdir,fname), 'w') as fw:
      fw.create_dataset('X', data=x)
      fw.create_dataset('Y', data=y)
      fw.create_dataset('Z', data=z)
      fw.create_dataset('DateTime', data=timestamp)
   
if __name__ == "__main__":
  main(sys.argv[1:])
