import numpy as np
import tensorflow.keras.backend as K

# Use custom metrics for training and evalution

# Recall
def micro_recall(y_true, y_pred):
  y_true_and_pred = K.cast(y_true * y_pred, 'float')
  true_pos = K.sum(y_true_and_pred, axis=0)
  possible_pos = K.sum(y_true, axis=0)
  rec = K.sum(true_pos) / (K.sum(possible_pos) + K.epsilon())
  return rec

def macro_recall(y_true, y_pred):
  y_true_and_pred = K.cast(y_true * y_pred, 'float')
  true_pos = K.sum(y_true_and_pred, axis=0)
  possible_pos = K.sum(y_true, axis=0)
  rec = K.mean(true_pos / (possible_pos + K.epsilon()))
  return rec

def weighted_recall(y_true, y_pred):
  y_true_and_pred = K.cast(y_true * y_pred, 'float')
  true_pos = K.sum(y_true_and_pred, axis=0)
  possible_pos = K.sum(y_true, axis=0)
  wts = K.sum(y_true, axis=0) / K.sum(y_true)
  rec = K.mean((true_pos * wts) / (possible_pos + K.epsilon()))
  return rec

# Precision
def micro_precision(y_true, y_pred):
  y_true_and_pred = K.cast(y_true * y_pred, 'float')
  true_pos = K.sum(y_true_and_pred, axis=0)
  pred_pos = K.sum(y_pred, axis=0)
  prec = K.sum(true_pos) / (K.sum(pred_pos) + K.epsilon())
  return prec

def macro_precision(y_true, y_pred):
  y_true_and_pred = K.cast(y_true * y_pred, 'float')
  true_pos = K.sum(y_true_and_pred, axis=0)
  pred_pos = K.sum(y_pred, axis=0)
  prec = K.mean(true_pos / (pred_pos + K.epsilon()))
  return prec

def weighted_precision(y_true, y_pred):
  y_true_and_pred = K.cast(y_true * y_pred, 'float')
  true_pos = K.sum(y_true_and_pred, axis=0)
  pred_pos = K.sum(y_pred, axis=0)
  wts = K.sum(y_true, axis=0) / K.sum(y_true)
  rec = K.mean((true_pos * wts) / (pred_pos + K.epsilon()))
  return rec

#F1 score

# get one-hot rep of given tensor
def get_one_hot(y):
  return K.one_hot(K.argmax(y,axis=1), y.shape[1])

def fbeta(prec, rec, beta=1.0):
  f_score = ((1+beta) * prec * rec)/(beta*prec + rec + K.epsilon())
  return f_score

def micro_f1(y_true, y_pred):
  y_pred = get_one_hot(y_pred)
  prec = micro_precision(y_true, y_pred)
  rec = micro_recall(y_true, y_pred)
  f1 = fbeta(prec, rec)
  return f1

def macro_f1(y_true, y_pred):
  y_pred = get_one_hot(y_pred)
  prec = macro_precision(y_true, y_pred)
  rec = macro_recall(y_true, y_pred)
  f1 = fbeta(prec, rec)
  return f1

def weighted_f1(y_true, y_pred):
  y_pred = get_one_hot(y_pred)
  prec = weighted_precision(y_true, y_pred)
  rec = weighted_recall(y_true, y_pred)
  f1 = fbeta(prec, rec)
  return f1

