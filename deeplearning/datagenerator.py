import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
import random
from tqdm import tqdm
from transforms import jitter, time_warp, rotation, rand_sampling
from transforms import get_ENMO, get_angle_z, get_LIDS
from collections import Counter

class DataGenerator(Sequence):
  def __init__(self, indices, data, labels, classes, partition=None, batch_size=32, seqlen=100, n_channels=3,
               n_classes=5, feat_channels=0, shuffle=False, augment=False, aug_factor=0.0, balance=False,
               mean=None, std=None):
    'Initialization'
    self.partition = partition
    self.seqlen = seqlen
    self.batch_size = batch_size
    self.indices = indices
    self.data = data
    self.labels = labels
    self.classes = classes
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.shuffle = shuffle
    self.augment = augment
    self.aug_factor = 0.0
    if self.augment == True:
      self.aug_factor = aug_factor
    self.aug_func = [jitter, time_warp, rotation, rand_sampling]
    self.balance = balance
    self.feat_channels = feat_channels
    self.mean = mean
    self.std = std
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    nbatches = int(len(self.indices)*(1.0+self.aug_factor) // self.batch_size)
    if (nbatches * self.batch_size) < (len(self.indices) * (1.0+self.aug_factor)):
      nbatches += 1
    return nbatches

  def __getitem__(self, index):
    'Generate one batch of data'
    if self.augment == True:
      assert self.aug_factor > 0.0
    aug_factor = 1.0 + self.aug_factor
    orig_sz = int(self.batch_size / aug_factor) # reduced batch size if aug_factor > 1
    # Generate indices of the batch
    if self.balance == False: # For inference
      st_idx = index*orig_sz
      if (index+1)*orig_sz <= (len(self.indices)-1):
        end_idx = (index+1)*orig_sz
      else:
        end_idx = len(self.indices)
      indices = self.indices[st_idx:end_idx]
    else: # Balance each minibatch to have same number of classes
      cls_sz = int((self.batch_size / aug_factor) // self.n_classes)
      if cls_sz * aug_factor * self.n_classes < self.batch_size:
        cls_sz += 1
      # Generate indices with balanced classes
      indices = []
      for cls in range(len(self.classes)):
        cls_idx = self.indices[self.labels[self.indices] == cls]
        indices.extend(np.random.choice(cls_idx,cls_sz,replace=False))
      # Choose batch sized indices
      random.shuffle(indices)
      indices = np.array(indices[:orig_sz])
      
    # Generate data
    X, y = self.__data_generation__(indices)

    return X, y

  def __data_generation__(self, indices):
    'Generates data containing batch_size samples'
    # X : (n_samples, *dim, n_channels)
    n_channels = self.n_channels + self.feat_channels
    if self.augment == True:
      # Initialization
      X = np.zeros((self.batch_size, self.seqlen, n_channels))
      y = np.ones((self.batch_size), dtype=int) * -1
  
      # Get number of samples per class after augmentation
      aug_factor = 1.0 + self.aug_factor
      cls_sz = self.batch_size // self.n_classes
      # number of samples from disk for each class
      samp_sz = [min(cls_sz, len(indices[self.labels[indices] == cls])) for cls in range(self.n_classes)]
      # number of samples to be augmented for each class
      aug_sz = [cls_sz - samp_sz[cls] for cls in range(self.n_classes)]
      sum_samp = sum(samp_sz) + sum(aug_sz)
      if sum_samp < self.batch_size:
        # Add more augmentation to smallest class
        idx = samp_sz.index(min(samp_sz))
        aug_sz[idx] += self.batch_size - sum_samp
      else:
        # Remove samples from largest class
        idx = samp_sz.index(max(samp_sz))
        aug_sz[idx] -= sum_samp - self.batch_size
      sum_samp = sum(samp_sz) + sum(aug_sz)
      assert sum_samp == self.batch_size

      # Generate data per class
      offset = 0
      for cls in range(self.n_classes):
        # Load data from disk
        cls_indices = indices[self.labels[indices] == cls][:samp_sz[cls]]
        for i, idx in enumerate(cls_indices):
          X[offset+i] = self.data[idx,:,:n_channels]
          y[offset+i] = self.labels[idx]
        # Choose a subset of samples to apply transformations for augmentation
        if aug_sz[cls] > 0:
          N = len(cls_indices)
          aug_indices = np.random.choice(cls_indices, aug_sz[cls], replace=True)
          aug_x = np.zeros((aug_sz[cls], self.seqlen, self.n_channels))
          for i,idx in enumerate(aug_indices):
            y[offset+N+i] = self.labels[idx]
            aug_x[i,] = self.data[idx,:,:self.n_channels]
          # Apply one or two transformations to x,y,z of the chosen data
          aug_x = random.choice(self.aug_func)(aug_x)
          toss = random.choice([0,1])
          if toss == 1:
            aug_x = random.choice(self.aug_func)(aug_x)
          # Get feature channels from augmented raw_data
          if self.feat_channels > 0:    
            ENMO = get_ENMO(aug_x[:,:,0], aug_x[:,:,1], aug_x[:,:,2])[:,:,np.newaxis]          
            angz = get_angle_z(aug_x[:,:,0], aug_x[:,:,1], aug_x[:,:,2])[:,:,np.newaxis]         
            LIDS = get_LIDS(aug_x[:,:,0], aug_x[:,:,1], aug_x[:,:,2])[:,:,np.newaxis] 
            aug_x = np.concatenate((aug_x, ENMO, angz, LIDS), axis=-1) 
        X[offset+N:offset+N+aug_sz[cls],:,:] = aug_x
        offset += samp_sz[cls] + aug_sz[cls]
      # Shuffle original and augmented data
      idx = np.arange(self.batch_size)
      np.random.shuffle(idx)
      X = X[idx]
      y = y[idx]
    else: 
      # Initialization
      X = np.zeros((len(indices), self.seqlen, n_channels))
      y = np.ones((len(indices)), dtype=int) * -1
      for i, idx in enumerate(indices):
        X[i] = self.data[idx,:,:n_channels]
        y[i] = self.labels[idx]

    # Normalize data if mean and std are present
    if self.mean is not None and self.std is not None:
      X = (X - self.mean) / self.std

    return X, to_categorical(y, num_classes=self.n_classes)
  
  def on_epoch_end(self):
    'Updates indexes after each epoch'
    if self.shuffle == True:
      np.random.shuffle(self.indices)

  def fit(self, frac=1.0):
    'Get mean and standard deviation for training data'
    assert 'stat' in self.partition
    samp_sum = np.zeros((self.seqlen, self.n_channels + self.feat_channels))
    samp_sqsum = np.zeros((self.seqlen, self.n_channels + self.feat_channels))
    N = int(frac*len(self)) # Get statistics using just a fraction of data
    nsamp = N*self.batch_size
    for i in tqdm(range(N)):
      X,y = self[i]
      samp_sum += X.sum(axis=0)
      samp_sqsum += (X**2).sum(axis=0)
    self.mean = samp_sum/nsamp
    self.std = np.sqrt(samp_sqsum/nsamp - self.mean**2)

    return self.mean, self.std
