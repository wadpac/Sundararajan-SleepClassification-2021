# -*- coding: utf-8 -*-
import sys
import h5py
import pandas as pd

def main(argv):
  infile1 = argv[0]
  infile2 = argv[1]
  infile3 = argv[2]
  outfile = argv[3]
  
  fp1 = h5py.File(infile1, 'r')
  rawdata1 = fp1['data']
  fp2 = h5py.File(infile2, 'r')
  rawdata2 = fp2['data']
  fp3 = h5py.File(infile3, 'r')
  rawdata3 = fp3['data']
  print(rawdata1.shape, rawdata2.shape, rawdata3.shape)
  
  with h5py.File(outfile, 'w') as fout:
    fout.create_dataset('data', data=rawdata1, compression='gzip', chunks=True,\
                        maxshape=(None,rawdata1.shape[1],rawdata1.shape[2]))
    fout['data'].resize((fout['data'].shape[0] + rawdata2.shape[0]), axis=0)
    fout['data'][-rawdata2.shape[0]:] = rawdata2
    fout['data'].resize((fout['data'].shape[0] + rawdata3.shape[0]), axis=0)
    fout['data'][-rawdata3.shape[0]:] = rawdata3
    print(fout['data'].shape)
    
if __name__ == "__main__":
    main(sys.argv[1:])

