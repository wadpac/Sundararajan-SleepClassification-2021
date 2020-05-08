import sys,os
import pandas as pd
import h5py
import numpy as np

def main(argv):
  partition_file = argv[0]
  feat_file = argv[1]
  rawdata_file = argv[2]
  outdir = argv[3]

  partition_df = pd.read_csv(partition_file)
  users = list(set(partition_df['user'].astype(str)))

  fname = os.path.basename(partition_file)
  partition = fname.split('_')[1]

  feat_df = pd.read_csv(feat_file)
  feat_df = feat_df[feat_df['user'].astype(str).isin(users)]
  indices = np.array(feat_df.index)
  feat_df = feat_df.reset_index(drop=True)
  feat_df.to_csv(os.path.join(outdir, fname), index=False)  

  fp = h5py.File(rawdata_file, 'r')
  rawdata = fp['data']

  fname = os.path.basename(rawdata_file)
  batchsz = 1000
  nbatches = len(indices)//batchsz
  if len(indices) % batchsz:
    nbatches += 1
  for i in range(nbatches):
    print('Processing %d/%d' % (i+1,nbatches))
    batch_indices = indices[i*batchsz:min((i+1)*batchsz, len(indices))]  
    if i == 0:
      with h5py.File(os.path.join(outdir, 'all_'+partition+'_'+fname), 'w') as fw:
        fw.create_dataset('data', data=rawdata[batch_indices], compression='gzip', chunks=True,\
                          maxshape=(None,rawdata.shape[1],rawdata.shape[2]))
    else:
      with h5py.File(os.path.join(outdir, 'all_'+partition+'_'+fname), 'a') as fw:
        fw['data'].resize((fw['data'].shape[0] + len(batch_indices)), axis=0)
        fw['data'][-len(batch_indices):] = rawdata[batch_indices]
      
  fp.close()

if __name__ == "__main__":
  main(sys.argv[1:])
