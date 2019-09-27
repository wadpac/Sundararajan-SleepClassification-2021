import numpy as np
import keras
import random
from tqdm import tqdm
from transforms import jitter, time_warp, rotation, rand_sampling

class DataGenerator(keras.utils.Sequence):
  def __init__(self, filenames, labels, classes, partition=None, batch_size=32, seqlen=100, n_channels=3,
               n_classes=5, mean=None, std=None, shuffle=False, augment=False, balance=False):
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
    self.aug_factor = 0.0
    self.aug_func = [jitter, time_warp, rotation, rand_sampling]
    self.balance = balance
    self.mean = mean
    self.std = std
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.filenames)*(1.0+self.aug_factor) / self.batch_size))

  def __getitem__(self, index):
    'Generate one batch of data'
    # Generate indices of the batch
    if self.balance == False: # For inference
      indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
    else: # Balance each minibatch to have same number of classes
      # Get number of indices from each class
      if self.augment == True:
        assert self.aug_factor > 0.0
      aug_factor = 1.0 + self.aug_factor
      cls_sz = (self.batch_size / aug_factor) // self.n_classes
      if cls_sz * aug_factor * self.n_classes < self.batch_size:
        cls_sz += 1
      # Generate indices with balanced classes
      indices = []
      all_indices = np.arange(len(self.filenames))
      for cls in self.classes:
        cls_idx = all_indices[self.labels == cls]
        indices.extend(np.random.choice(cls_idx,cls_sz))          
      # Choose batch sized indices
      random.shuffle(indices)
      indices = indices[:self.batch_size]

    # Generate data
    X, y = self.__data_generation(indices)

    return X, y

  def __data_generation(self, indices):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    X = np.empty((self.batch_size, self.seqlen, self.n_channels))
    y = np.empty((self.batch_size), dtype=int)

    # Generate data
    for i, idx in enumerate(indices):
      X[i,] = np.load(self.filenames[idx])
      y[i] = self.labels[idx]

    # Augment data
    if self.augment == True:
      # Choose a subset of samples to apply transformations for augmentation
      N = len(indices)
      n_aug = self.batchsize - N
      aug_indices = np.random.choice(indices, n_aug)
      aug_x = np.empty((n_aug, self.seqlen, self.n_channels))
      for i,idx in enumerate(aug_indices):
        y[N+i] = self.labels[idx]
        aug_x[i,] = np.load(self.filenames[idx])
      # Apply one or two transformations to the chosen data
      aug_x = random.choice(variant_func)(lbl_x[rand_idx])
      toss = random.choice([0,1])
      if toss == 1:
        aug_x = random.choice(variant_func)(aug_x)
      X[N:,] = aug_x

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
    mean = np.empty((self.seqlen, self.n_channels))
    std = np.empty((self.seqlen, self.n_channels))
    N = len(self.filenames)
    for i in tqdm(range(N)):
      mean += np.load(self.filenames[i])
    self.mean = mean/N
    for i in tqdm(range(N)):
      x = np.load(self.filenames[i])
      std += (x - mean)**2
    self.std = std/N
    return self.mean, self.std
