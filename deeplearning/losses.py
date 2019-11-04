from keras import backend as K

def weighted_categorical_crossentropy(weights):
  """
  Weighted categorical cross-entropy for unbalanced datasets

  Variables:
    weights: numpy array of shape (num_classes,)
  """
  weights = K.variable(weights)
  epsilon = K.epsilon()

  def loss(y_true, y_pred):
    y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, axis=-1)
    return loss 

  return loss

  
def focal_loss(gamma=2.0, alpha=1.0):
  """
  Focal loss for unbalanced datasets

  Variables:
    gamma: focus parameter to emphasize hard examples
    alpha: numpy array of shape (num_classes,) denoting class weights
  """
  alpha = K.variable(alpha)
  epsilon = K.epsilon()

  def loss(y_true, y_pred):
    y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
    ce_loss = y_true * K.log(y_pred)
    focal_loss = K.pow(1 - y_pred, gamma) * ce_loss * alpha
    focal_loss = -K.sum(focal_loss, axis=-1)
    return focal_loss 

  return loss
