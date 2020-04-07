import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.callbacks import Callback

# Use custom callback for F1-score, precision and recall
class Metrics(Callback):
  def __init__(self, val_data, batch_size=32, min_epoch=5):
    self.validation_data = val_data # validation generator  
    self.batch_size = batch_size
    self.min_epoch = min_epoch
    self.val_precision = []
    self.val_recall = []
    self.val_f1 = []

  def on_train_begin(self, logs=None):
    self.val_precision = []
    self.val_recall = []
    self.val_f1 = []
 
  # Get predictions given a data generator
  def get_predictions(self, datagen):
    batches = len(datagen)
    nsamples = batches * self.batch_size
    y_true = np.zeros((nsamples,))
    y_pred = np.zeros((nsamples,))
    actual_nsamples = 0
    for batch in range(batches):
      x, y = datagen[batch]
      end_idx = batch * self.batch_size + y.shape[0]
      y_pred[batch * self.batch_size : end_idx] = self.model.predict(x).argmax(axis=1)
      y_true[batch * self.batch_size : end_idx] = y.argmax(axis=1)
      actual_nsamples += y.shape[0]
    y_true = y_true[:actual_nsamples]
    y_pred = y_pred[:actual_nsamples]
    return y_true, y_pred

  def on_epoch_end(self, epoch, logs=None):
    # Get predictions on validation data
    val_true, val_pred = self.get_predictions(self.validation_data)
    val_prec, val_rec, val_f1, val_sup = \
            precision_recall_fscore_support(val_true, val_pred, average='macro')
    self.val_precision.append(val_prec)
    self.val_recall.append(val_rec)
    self.val_f1.append(val_f1)
    if epoch >= self.min_epoch: # save validation metrics only after first few epochs
      logs['val_f1'] = val_f1
    else:
      logs['val_f1'] = 0.0

    print(' - val_f1: {:0.4f}'.format(val_f1))

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
