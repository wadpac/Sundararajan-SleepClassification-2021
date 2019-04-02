# -*- coding: utf-8 -*-
import sys, os
import pandas as pd
import pickle

def main(argv):
  indir1 = argv[0]
  indir2 = argv[1]
  outfile = argv[2]
    
  features = []; labels = []; users = []

  files = os.listdir(indir1)
  for fname in files:
    data = pickle.load(open(os.path.join(indir1,fname),'rb'))
    features.extend([data[idx]['features'] for idx in range(len(data))])
    labels.extend([data[idx]['labels'] for idx in range(len(data))])
    users.extend([data[idx]['user'] for idx in range(len(data))])
 
  files = os.listdir(indir2)
  for fname in files:
    data = pickle.load(open(os.path.join(indir2,fname),'rb'))
    features.extend([data[idx]['features'] for idx in range(len(data))])
    labels.extend([data[idx]['labels'] for idx in range(len(data))])
    users.extend([data[idx]['user'] for idx in range(len(data))])

  data = {'features': features, 'labels': labels, 'users': users}
  pickle.dump(data, open(outfile,'wb'))
 
if __name__ == "__main__":
  main(sys.argv[1:])

