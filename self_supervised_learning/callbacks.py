import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.callbacks import Callback

class BatchRenormScheduler(Callback):
  def __init__(self, epoch_batches, renorm_momentum=0.99):
    self.batch = 0
    self.rmax = 1
    self.dmax = 0
    self.epoch_batches = epoch_batches

  def on_train_begin(self, logs=None):
    self.batch = 0
    self.rmax = 1
    self.dmax = 0

  def on_train_batch_end(self, batch, logs=None):
    self.batch += 1
    if self.batch % (self.epoch_batches//2) == 0: # Gradually relax batch renorm parameters
      self.rmax += 0.25
      self.dmax += 1
    for layer in self.model.layers:
      if layer.name.startswith('bn'):
        if self.batch < self.epoch_batches: # Batch normalization for first epoch
          layer.renorm = False
        elif self.batch % (self.epoch_batches//2) == 0: # Update batch renorm parameters
          layer.renorm = True
          layer.renorm_clipping = {'rmin':1.0/min(2,self.rmax), 'rmax':min(2,self.rmax), 'dmax':min(4,self.dmax)}
          layer.renorm_momentum = 0.99
