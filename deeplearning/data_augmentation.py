import os
import random
import numpy as np
from collections import Counter
from transforms import jitter, time_warp, rotation, rand_sampling

# augment data - aug_factor : factor by which majority class samples are augmented.
# Samples from all other classes will be augmented by same amount
def augment(X, y, sleep_states, aug_factor=1.0, step_sz = 10000):
  if not os.path.exists('tmp'):
    os.makedirs('tmp')

  # List of possible variations
  variant_func = [jitter, time_warp, rotation, rand_sampling]

  # Merge wake and extra_wake
  wake_ext_idx = sleep_states.index('Wake_ext')
  wake_idx = sleep_states.index('Wake')
  y[y[:,wake_ext_idx] == 1,wake_idx] = 1
  y = np.hstack((y[:,:wake_ext_idx], y[:,wake_ext_idx+1:-1]))
  sleep_states = [state for state in sleep_states if state != 'Wake_ext']
  print(sleep_states)

  y_lbl = y.argmax(axis=1)
  y_lbl = [sleep_states[i] for i in y_lbl]
  y_ctr = Counter(y_lbl).most_common()
  print(y_ctr)

  # Find max number of samples for each class
  # Create memory mapped arrays to store data after augmentation
  max_samp = int(y_ctr[0][1]*aug_factor)
  naug_samp = max_samp * len(sleep_states)
  X_aug = np.memmap('tmp/X_aug.np', dtype='float32', mode='w+', shape=(naug_samp,X.shape[1],X.shape[2]))
  y_aug = np.memmap('tmp/y_aug.np', dtype='int32', mode='w+', shape=(naug_samp,y.shape[1]))
  for i,state in enumerate(y_ctr):
    lbl,ctr = state
    idx = sleep_states.index(lbl)
    lbl_y = y[y[:,idx] == 1,:]
    lbl_x = X[y[:,idx] == 1,:]
    st_idx = i*max_samp

    # Augment each class with random variations
    # Choose a random set of samples and apply a single variation
    # Toss a coin and choose to apply another variation
    # Append new samples to augmented data
    n_aug = max_samp - ctr
    if n_aug > 0:
      print('%s: Augmenting %d samples to %d samples' % (lbl,n_aug,ctr))
      end_idx = st_idx + lbl_x.shape[0]
      X_aug[st_idx:end_idx,:,:] = lbl_x
      y_aug[st_idx:end_idx,:] = lbl_y
      if n_aug <= lbl_x.shape[0]: # Few samples to be augmented
        rand_idx = np.random.randint(0,lbl_x.shape[0],n_aug)
        aug_x = random.choice(variant_func)(lbl_x[rand_idx])
        toss = random.choice([0,1])
        if toss == 1:
          aug_x = random.choice(variant_func)(aug_x)
        st_idx = end_idx
        end_idx = st_idx + aug_x.shape[0]
        X_aug[st_idx:end_idx,:,:] = aug_x
        aug_y = np.zeros((n_aug,len(sleep_states))); aug_y[:,idx] = 1
        y_aug[st_idx:end_idx,:] = aug_y
      else: # If more samples to be augmented, create variations in batches of step_sz
        aug_x = None; aug_y = None
        for j in range(0,n_aug,step_sz):
          if j+step_sz < n_aug:
              sz = step_sz
          else:
              sz = n_aug-j
          rand_idx = np.random.randint(0,lbl_x.shape[0],sz)
          x_var = random.choice(variant_func)(lbl_x[rand_idx])
          toss = random.choice([0,1])
          if toss == 1:
            x_var = random.choice(variant_func)(x_var)
          y_var = np.zeros((sz,len(sleep_states))); y_var[:,idx] = 1
          if aug_x is not None:
            aug_x = np.vstack((aug_x,x_var))
            aug_y = np.vstack((aug_y,y_var))
          else:
            aug_x = x_var
            aug_y = y_var
        st_idx = end_idx
        end_idx = st_idx + aug_x.shape[0]
        X_aug[st_idx:end_idx,:,:] = aug_x
        y_aug[st_idx:end_idx,:] = aug_y
    else:
      print('%s: Choosing %d samples of %d samples' % (lbl,max_samp,ctr))
      end_idx = st_idx + max_samp
      rand_idx = np.random.randint(0,lbl_x.shape[0],max_samp)
      X_aug[st_idx:end_idx,:,:] = lbl_x[rand_idx]
      y_aug[st_idx:end_idx,:] = lbl_y[rand_idx]


  # Shuffle indices
  shuf_idx = np.arange(X_aug.shape[0])
  np.random.shuffle(shuf_idx)
  X_aug = X_aug[shuf_idx]
  y_aug = y_aug[shuf_idx]

  y_lbl = y_aug.argmax(axis=1)
  y_lbl = [sleep_states[i] for i in y_lbl]
  y_ctr = Counter(y_lbl).most_common()
  print(y_ctr)

  # Flush and close memmap objects
  del(X_aug); del(y_aug)

  return naug_samp
