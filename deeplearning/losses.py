from keras import backend as K

def weighted_categorical_crossentropy(weights):
  """
  Weighted categorical cross-entropy for unbalanced datasets

  Variables:
    weights: numpy array of shape (num_classes,)
  """
  weights = K.variable(weights)

  def loss(y_true, y_pred):
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, axis=-1)
    return loss 

  return loss
