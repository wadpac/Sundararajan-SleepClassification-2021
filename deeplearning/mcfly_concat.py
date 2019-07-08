import sys,os
import numpy as np
from scipy.interpolate import interp1d

def main(argv):
  indir = argv[0]
  seqlen = int(argv[1]) # usually resampled frequency (30-40Hz) * time interval
  outfile = argv[2]

  files = os.listdir(indir)
 
  all_data = None
  all_labels = None
  all_users = [] 
  all_dataset = []

  for fname in files:
    fdata = np.load(os.path.join(indir,fname))
    data = fdata['data']
    # Resample data
    orig_seqlen = data.shape[1]
    spacing = orig_seqlen/float(seqlen)
    data[:,:seqlen,0] = interp1d(np.arange(0,orig_seqlen), data[:,:,0])(np.arange(0,orig_seqlen,spacing))
    data[:,:seqlen,1] = interp1d(np.arange(0,orig_seqlen), data[:,:,1])(np.arange(0,orig_seqlen,spacing))
    data[:,:seqlen,2] = interp1d(np.arange(0,orig_seqlen), data[:,:,2])(np.arange(0,orig_seqlen,spacing))
    data = data[:,:seqlen,:]
    labels = fdata['labels']
    if all_data is None:
      all_data = data
      all_labels = labels
    else:
      all_data = np.concatenate((all_data,data))
      all_labels = np.concatenate((all_labels,labels))
    user = str(fdata['user'])
    dataset = str(fdata['dataset'])
    all_users.extend([user]*data.shape[0])
    all_dataset.extend([dataset]*data.shape[0])

  print(all_data.shape, all_labels.shape)
  np.savez(outfile, data=all_data, labels=all_labels, user=all_users, dataset=all_dataset)

if __name__ == "__main__":
  main(sys.argv[1:])
