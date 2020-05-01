import sys,os
import numpy as np
import random
import h5py
from tqdm import tqdm
from collections import Counter

def save_partition(infile, indices, partition, outdir, batchsize=1000):
  with h5py.File(infile,'r') as fp:
    samples1 = fp['samp1']
    samples2 = fp['samp2']
    labels = fp['label']
    [num_samples, seqlen, channels] = samples1.shape
    
    # Save partition
    num_indices = len(indices)
    num_batches = num_indices // batchsize
    if num_batches % batchsize:
      num_batches += 1
    for i in tqdm(range(num_batches)):
      batch_indices = indices[i*batchsize:min(num_indices,(i+1)*batchsize)]  
      if i == 0:
        with h5py.File(os.path.join(outdir, partition+'_dataset.h5'),'w') as fp:
          fp.create_dataset('samp1', data=samples1[batch_indices], chunks=True,\
                            compression='gzip', maxshape=(None,seqlen,channels))
          fp.create_dataset('samp2', data=samples2[batch_indices], chunks=True,\
                            compression='gzip', maxshape=(None,seqlen,channels))
          fp.create_dataset('label', data=labels[batch_indices], chunks=True,\
                            compression='gzip', maxshape=(None,))
      else:
        with h5py.File(os.path.join(outdir, partition+'_dataset.h5'),'a') as fp:
          fp['samp1'].resize((fp['samp1'].shape[0] + len(batch_indices)), axis=0)
          fp['samp1'][-len(batch_indices):] = samples1[batch_indices]
          fp['samp2'].resize((fp['samp2'].shape[0] + len(batch_indices)), axis=0)
          fp['samp2'][-len(batch_indices):] = samples2[batch_indices]
          fp['label'].resize((fp['label'].shape[0] + len(batch_indices)), axis=0)
          fp['label'][-len(batch_indices):] = labels[batch_indices]

def main(args):
  infile = args[0]
  val_perc = float(args[1])
  test_perc = float(args[2])
  outdir = args[3]

  with h5py.File(infile,'r') as fp:
    samples1 = fp['samp1']
    samples2 = fp['samp2']
    labels = fp['label']
    [num_samples, seqlen, channels] = samples1.shape

    num_val = int(val_perc/100.0 * num_samples)
    num_test = int(test_perc/100.0 * num_samples)
    num_train = num_samples - num_val - num_test

    indices = np.arange(num_samples)
    random.shuffle(indices)
    train_indices = np.sort(indices[:num_train])
    val_indices = np.sort(indices[num_train:num_train+num_val])
    test_indices = np.sort(indices[num_train+num_val:])

    # Save partitions
    batchsize = 100
    save_partition(infile, train_indices, 'train', outdir, batchsize=batchsize)
    save_partition(infile, val_indices, 'val', outdir, batchsize=batchsize)
    save_partition(infile, test_indices, 'test', outdir, batchsize=batchsize)

if __name__ == "__main__":
  main(sys.argv[1:])
