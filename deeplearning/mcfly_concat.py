import sys,os
import numpy as np
from collections import Counter

def main(argv):
  indir1 = argv[0]
  indir2 = argv[1]
  seqlen = int(argv[2])
  outfile = argv[3]

  files1 = os.listdir(indir1)
  files2 = os.listdir(indir2)
 
  all_data = None
  all_labels = None
  all_users = [] 

  for fname in files1:
    fdata = np.load(os.path.join(indir1,fname))
    data = fdata['data']
    prev_seqlen = data.shape[1]
    data.resize((data.shape[0], seqlen, data.shape[2]), refcheck=False)
    data[:,prev_seqlen:,:] = 0
    labels = fdata['labels']
    if all_data is None:
      all_data = data
      all_labels = labels
    else:
      all_data = np.concatenate((all_data,data))
      all_labels = np.concatenate((all_labels,labels))
    user = str(fdata['user'])
    all_users.extend([user]*data.shape[0])

  for fname in files2:
    fdata = np.load(os.path.join(indir2,fname))
    data = fdata['data']
    prev_seqlen = data.shape[1]
    data.resize((data.shape[0], seqlen, data.shape[2]), refcheck=False)
    data[:,prev_seqlen:,:] = 0
    all_data = np.concatenate((all_data,data))
    labels = fdata['labels']
    all_labels = np.concatenate((all_labels,labels))
    user = str(fdata['user'])
    all_users.extend([user]*data.shape[0])

  np.savez(outfile, data=all_data, labels=all_labels, user=all_users)

if __name__ == "__main__":
  main(sys.argv[1:])
