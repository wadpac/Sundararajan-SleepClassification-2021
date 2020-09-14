import kerastuner
import numpy as np
from sklearn.model_selection import GroupKFold
from datagenerator import DataGenerator
from collections import Counter

class CVTuner(kerastuner.engine.tuner.Tuner):

  def __init__(self, cv=5, states=None, num_classes=None, seqlen=None, num_channels=None,\
               feat_channels=None, mean=None, std=None, *args, **kwargs):
    super(CVTuner, self).__init__(*args, **kwargs)
    self.cv = cv
    self.states = states
    self.num_classes = num_classes
    self.seqlen = seqlen
    self.num_channels = num_channels
    self.feat_channels = feat_channels
    self.mean = mean
    self.std = std
    self.ntrial = 0

  def run_trial(self, trial, data=None, labels=None, users=None, indices=None, batch_size=32):
    self.ntrial += 1
    hp = trial.hyperparameters
    print('Trial {:d}'.format(self.ntrial))
    print(hp.values)
    if "tuner/trial_id" in hp:
      past_trial = self.oracle.get_trial(hp['tuner/trial_id'])
      model = self.load_model(past_trial)
    else:
      model = self.hypermodel.build(hp)
    initial_epoch = hp['tuner/initial_epoch']
    epochs = hp['tuner/epochs']

    # Cross-validation based on users
    fold = 0
    val_losses = []
    cv = GroupKFold(n_splits=self.cv)
    trial_users = users[indices]
    X = np.zeros((len(trial_users),10)); y = np.zeros(len(trial_users)) # dummy for splitting
    for train_indices, val_indices in cv.split(X, y, trial_users):
      fold += 1
      print('Inner CV fold {:d}'.format(fold))
      train_gen = DataGenerator(indices[train_indices], data, labels, self.states, partition='train',\
                                batch_size=batch_size, seqlen=self.seqlen, n_channels=self.num_channels, feat_channels=self.feat_channels,\
                                n_classes=self.num_classes, shuffle=True, balance=True, mean=self.mean, std=self.std)
      val_gen = DataGenerator(indices[val_indices], data, labels, self.states, partition='test',\
                              batch_size=batch_size, seqlen=self.seqlen, n_channels=self.num_channels, feat_channels=self.feat_channels,\
                              n_classes=self.num_classes, mean=self.mean, std=self.std)
      model = self.hypermodel.build(trial.hyperparameters)
      model.fit(train_gen, epochs=epochs, validation_data=val_gen,\
                verbose=1, shuffle=False, initial_epoch=initial_epoch,
                workers=2, max_queue_size=20, use_multiprocessing=False )
      val_losses.append(model.evaluate(val_gen))
    self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(val_losses)})
    self.save_model(trial.trial_id, model)
