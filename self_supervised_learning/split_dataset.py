import sys,os
import numpy as np
import random

def main(args):
  sampfile1 = args[0]
  sampfile2 = args[1]
  lblfile = args[2]
  val_perc = float(args[3])
  test_perc = float(args[4])
  outdir = args[5]

  shape_str = sampfile1.split('.npy')[0].split('_')[1]
  [num_samples, seqlen, channels] = [int(elem) for elem in shape_str.split('x')]

  num_val = int(val_perc/100.0 * num_samples)
  num_test = int(test_perc/100.0 * num_samples)
  num_train = num_samples - num_val - num_test

  indices = np.arange(num_samples)
  random.shuffle(indices)
  train_indices = indices[:num_train]
  val_indices = indices[num_train:num_train+num_val]
  test_indices = indices[num_train+num_val:]

  samples1 = np.memmap(sampfile1, dtype=np.float32, mode='r', shape=(num_samples, seqlen, channels))  
  samples2 = np.memmap(sampfile2, dtype=np.float32, mode='r', shape=(num_samples, seqlen, channels))  
  labels = np.memmap(lblfile, dtype=np.int32, mode='r', shape=(num_samples,))  

  # Get train data
  train_str = 'x'.join(elem for elem in [str(num_train), str(seqlen), str(channels)])
  train_samples1 = np.memmap(os.path.join(outdir, 'train_samples1_'+train_str+'.npy'), dtype=np.float32, mode='w+',
                             shape=(num_train, seqlen, channels))
  train_samples2 = np.memmap(os.path.join(outdir, 'train_samples2_'+train_str+'.npy'), dtype=np.float32, mode='w+',
                             shape=(num_train, seqlen, channels))
  train_labels = np.memmap(os.path.join(outdir, 'train_labels_'+train_str+'.npy'), dtype=np.int32, mode='w+',
                             shape=(num_train,))
  train_samples1[:,:,:] = samples1[train_indices,:,:]
  train_samples2[:,:,:] = samples2[train_indices,:,:]
  train_labels[:] = labels[train_indices]
  del(train_samples1)
  del(train_samples2)
  del(train_labels)

  # Get validation data
  val_str = 'x'.join(elem for elem in [str(num_val), str(seqlen), str(channels)])
  val_samples1 = np.memmap(os.path.join(outdir, 'val_samples1_'+val_str+'.npy'), dtype=np.float32, mode='w+',
                             shape=(num_val, seqlen, channels))
  val_samples2 = np.memmap(os.path.join(outdir, 'val_samples2_'+val_str+'.npy'), dtype=np.float32, mode='w+',
                             shape=(num_val, seqlen, channels))
  val_labels = np.memmap(os.path.join(outdir, 'val_labels_'+val_str+'.npy'), dtype=np.int32, mode='w+',
                             shape=(num_val,))
  val_samples1[:,:,:] = samples1[val_indices,:,:]
  val_samples2[:,:,:] = samples2[val_indices,:,:]
  val_labels[:] = labels[val_indices]
  del(val_samples1)
  del(val_samples2)
  del(val_labels)

  # Get test data
  test_str = 'x'.join(elem for elem in [str(num_test), str(seqlen), str(channels)])
  test_samples1 = np.memmap(os.path.join(outdir, 'test_samples1_'+test_str+'.npy'), dtype=np.float32, mode='w+',
                             shape=(num_test, seqlen, channels))
  test_samples2 = np.memmap(os.path.join(outdir, 'test_samples2_'+test_str+'.npy'), dtype=np.float32, mode='w+',
                             shape=(num_test, seqlen, channels))
  test_labels = np.memmap(os.path.join(outdir, 'test_labels_'+test_str+'.npy'), dtype=np.int32, mode='w+',
                             shape=(num_test,))
  test_samples1[:,:,:] = samples1[test_indices,:,:]
  test_samples2[:,:,:] = samples2[test_indices,:,:]
  test_labels[:] = labels[test_indices]
  del(test_samples1)
  del(test_samples2)
  del(test_labels)

if __name__ == "__main__":
  main(sys.argv[1:])
