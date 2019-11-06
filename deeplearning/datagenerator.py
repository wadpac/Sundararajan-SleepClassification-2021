import numpy as np
import keras
import random
from tqdm import tqdm
from transforms import jitter, time_warp, rotation, rand_sampling
from collections import Counter

class DataGenerator(keras.utils.Sequence):
  def __init__(self, filenames, labels, classes, partition=None, batch_size=32, seqlen=100, n_channels=3,
               n_classes=5, mean=None, std=None, shuffle=False, augment=False, aug_factor=0.0, balance=False):
    'Initialization'
    self.partition = partition
    self.seqlen = seqlen
    self.batch_size = batch_size
    self.labels = labels
    self.classes = classes
    self.filenames = filenames
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.shuffle = shuffle
    self.augment = augment
    self.aug_factor = aug_factor
    self.aug_func = [jitter, time_warp, rotation, rand_sampling]
    self.balance = balance
    self.mean = mean
    self.std = std
    self.on_epoch_end()
    self.selected = []

  def __len__(self):
    'Denotes the number of batches per epoch'
    nbatches = int(len(self.filenames)*(1.0+self.aug_factor) // self.batch_size)
    if (nbatches * self.batch_size) < (len(self.filenames) * (1.0+self.aug_factor)):
      nbatches += 1
    return nbatches

  def __getitem__(self, index):
    'Generate one batch of data'
    # Generate indices of the batch
    if self.balance == False: # For inference
      st_idx = index*self.batch_size
      if (index+1)*self.batch_size <= (len(self.filenames)-1):
        end_idx = (index+1)*self.batch_size
      else:
        end_idx = len(self.filenames)
      indices = self.indices[st_idx:end_idx]
    else: # Balance each minibatch to have same number of classes
      # Get indices from all classes as per data distribution
      # Augment samples to balance classes
      if self.augment == True:
        assert self.aug_factor > 0.0
      aug_factor = 1.0 + self.aug_factor
      # Generate indices with same distributin as data
      orig_sz = int(self.batch_size/aug_factor)
      all_indices = np.arange(len(self.filenames))
      # Ensure each class has atleast one sample
      samp_sz = [0] * self.n_classes
      while 0 in samp_sz:
        indices = np.random.choice(all_indices, orig_sz, replace=False)
        samp_sz = [len(indices[self.labels[indices] == cls]) for cls in range(self.n_classes)]
      # Choose batch sized indices
      random.shuffle(indices)
      indices = indices[:orig_sz]
      
    # Generate data
    X, y = self.__data_generation(indices)

    return X, y

  def __data_generation(self, indices):
    'Generates data containing batch_size samples'
    # X : (n_samples, *dim, n_channels)
    if self.augment == True:
      # Initialization
      X = np.zeros((self.batch_size, self.seqlen, self.n_channels))
      y = np.ones((self.batch_size), dtype=int) * -1
  
      # Get number of samples per class after agumentation
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
          X[offset+i,] = np.load(self.filenames[idx])
          y[offset+i] = self.labels[idx]
        # Choose a subset of samples to apply transformations for augmentation
        if aug_sz[cls] > 0:
          N = len(cls_indices)
          aug_indices = np.random.choice(cls_indices, aug_sz[cls], replace=True)
          aug_x = np.zeros((aug_sz[cls], self.seqlen, self.n_channels))
          for i,idx in enumerate(aug_indices):
            y[offset+N+i] = self.labels[idx]
            aug_x[i,] = np.load(self.filenames[idx])
          # Apply one or two transformations to the chosen data
          aug_x = random.choice(self.aug_func)(aug_x)
          toss = random.choice([0,1])
          if toss == 1:
            aug_x = random.choice(self.aug_func)(aug_x)
          X[offset+N:offset+N+aug_sz[cls],:,:] = aug_x
        offset += samp_sz[cls] + aug_sz[cls]
      # Shuffle original and augmented data
      idx = np.arange(self.batch_size)
      np.random.shuffle(idx)
      X = X[idx]
      y = y[idx]
    else: 
      # Initialization
      X = np.zeros((len(indices), self.seqlen, self.n_channels))
      y = np.ones((len(indices)), dtype=int) * -1
  
      for i, idx in enumerate(indices):
        X[i,] = np.load(self.filenames[idx])
        y[i] = self.labels[idx]

    # Normalize data if mean and std are present
    if self.mean is not None and self.std is not None:
      X = (X - self.mean)/self.std

    return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
  
  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indices = np.arange(len(self.filenames))
    if self.shuffle == True:
      np.random.shuffle(self.indices)

  def fit(self):
    'Get mean and std deviation for training data'
    assert 'train' in self.partition
    
    mean = np.zeros((self.seqlen, self.n_channels))
    std = np.zeros((self.seqlen, self.n_channels))

    N = len(self)
    for i in tqdm(range(N)):
      X,y = self[i]
      mean += X.sum(axis=0)
    self.mean = mean/len(self.filenames)
    for i in tqdm(range(N)):
      X,y = self[i]
      std += ((X - mean)**2).sum(axis=0)
    self.std = std/len(self.filenames)
    return self.mean, self.std
