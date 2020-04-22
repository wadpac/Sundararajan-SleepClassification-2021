import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
import random
from transforms import jitter, time_warp, rotation, rand_sampling
from transforms import get_ENMO, get_angle_z, get_LIDS

class DataGenerator(Sequence):
  def __init__(self, samples1, samples2, labels, classes=2, batch_size=32, seqlen=100, channels=3,\
               shuffle=False, balance=False, augment=False, aug_factor=0.0):
    'Initialization'
    self.seqlen = seqlen
    self.batch_size = batch_size
    self.classes = classes
    self.samples1 = samples1
    self.samples2 = samples2
    self.labels = labels
    self.channels = channels
    self.shuffle = shuffle
    self.balance = balance
    self.augment = augment
    self.aug_factor = 0.0
    if self.augment == True:
      self.aug_factor = aug_factor
    self.aug_func = [jitter, time_warp, rotation, rand_sampling]
    self.indices = np.arange(len(labels))
    if self.shuffle == True:
      np.random.shuffle(self.indices)

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
      cls_sz = int((self.batch_size / aug_factor) // self.classes)
      if cls_sz * aug_factor * self.classes < self.batch_size:
        cls_sz += 1
      # Generate indices with balanced classes
      indices = []
      for cls in range(self.classes):
        cls_idx = self.indices[self.labels[self.indices] == cls]
        indices.extend(np.random.choice(cls_idx,cls_sz,replace=False))
      # Choose batch sized indices
      random.shuffle(indices)
      indices = np.array(indices[:orig_sz])
    
    # Generate data
    X1, X2, y = self.__data_generation__(indices)
    
    return (X1, X2), y

  def __data_generation__(self, indices):
    'Generates data containing batch_size samples'
    # X : (n_samples, *dim, channels)
    if self.augment == True:
      # Initialization
      X1 = np.zeros((self.batch_size, self.seqlen, self.channels))
      X2 = np.zeros((self.batch_size, self.seqlen, self.channels))
      y = np.ones((self.batch_size), dtype=int) * -1
  
      # Get number of samples per class after augmentation
      aug_factor = 1.0 + self.aug_factor
      cls_sz = self.batch_size // self.classes
      # number of samples from disk for each class
      samp_sz = [min(cls_sz, len(indices[self.labels[indices] == cls])) for cls in range(self.classes)]
      # number of samples to be augmented for each class
      aug_sz = [cls_sz - samp_sz[cls] for cls in range(self.classes)]
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
      for cls in range(self.classes):
        # Load data from disk
        cls_indices = indices[self.labels[indices] == cls][:samp_sz[cls]]
        for i, idx in enumerate(cls_indices):
          X1[offset+i] = self.samples1[idx,:,:self.channels]
          X2[offset+i] = self.samples2[idx,:,:self.channels]
          y[offset+i] = self.labels[idx]
        # Choose a subset of samples to apply transformations for augmentation
        if aug_sz[cls] > 0:
          N = len(cls_indices)
          aug_indices = np.random.choice(cls_indices, aug_sz[cls], replace=True)
          aug_x1 = np.zeros((aug_sz[cls], self.seqlen, 3))
          aug_x2 = np.zeros((aug_sz[cls], self.seqlen, 3))
          for i,idx in enumerate(aug_indices):
            y[offset+N+i] = self.labels[idx]
            aug_x1[i,] = self.samples1[idx,:,:3]
            aug_x2[i,] = self.samples2[idx,:,:3]
          # Apply one or two transformations to x,y,z of the chosen data
          aug_x1 = random.choice(self.aug_func)(aug_x1)
          aug_x2 = random.choice(self.aug_func)(aug_x2)
          toss = random.choice([0,1])
          if toss == 1:
            aug_x1 = random.choice(self.aug_func)(aug_x1)
            aug_x2 = random.choice(self.aug_func)(aug_x2)
          # Get feature channels from augmented raw_data
          if self.channels > 3:    
            ENMO = get_ENMO(aug_x1[:,:,0], aug_x1[:,:,1], aug_x1[:,:,2])[:,:,np.newaxis]          
            angz = get_angle_z(aug_x1[:,:,0], aug_x1[:,:,1], aug_x1[:,:,2])[:,:,np.newaxis]         
            LIDS = get_LIDS(aug_x1[:,:,0], aug_x1[:,:,1], aug_x1[:,:,2])[:,:,np.newaxis] 
            aug_x1 = np.concatenate((aug_x1, ENMO, angz, LIDS), axis=-1) 
            ENMO = get_ENMO(aug_x2[:,:,0], aug_x2[:,:,1], aug_x2[:,:,2])[:,:,np.newaxis]          
            angz = get_angle_z(aug_x2[:,:,0], aug_x2[:,:,1], aug_x2[:,:,2])[:,:,np.newaxis]         
            LIDS = get_LIDS(aug_x2[:,:,0], aug_x2[:,:,1], aug_x2[:,:,2])[:,:,np.newaxis] 
            aug_x2 = np.concatenate((aug_x2, ENMO, angz, LIDS), axis=-1) 
          X1[offset+N:offset+N+aug_sz[cls],:,:] = aug_x1
          X2[offset+N:offset+N+aug_sz[cls],:,:] = aug_x2
          offset += aug_sz[cls]
        offset += samp_sz[cls]

      # Shuffle original and augmented data
      idx = np.arange(self.batch_size)
      np.random.shuffle(idx)
      X1 = X1[idx]
      X2 = X2[idx]
      y = y[idx]
    else: 
      # Initialization
      X1 = self.samples1[indices,:,:]
      X2 = self.samples2[indices,:,:]
      y = self.labels[indices]

    return X1, X2, y
  
  def on_epoch_end(self):
    'Updates indexes after each epoch'
    if self.shuffle == True:
      np.random.shuffle(self.indices)
