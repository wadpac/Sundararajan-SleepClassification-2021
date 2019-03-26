import sys,os
import numpy as np
import pandas as pd
from mcfly import modelgen, find_architecture, storage
from keras.models import load_model

from sklearn.model_selection import GroupKFold, StratifiedKFold, RandomizedSearchCV

np.random.seed(2)

def main(argv):
  infile = argv[0]
  outdir = argv[1]

  if not os.path.exists(outdir):
    os.makedirs(outdir)

  resultdir = os.path.join(outdir,'models')
  if not os.path.exists(resultdir):
    os.makedirs(resultdir)

  all_data = np.load(infile)
  X = all_data['data']
  y = all_data['labels']
  users = all_data['user']

  # Use nested cross-validation based on users
  # Outer CV
  outer_cv_splits = 5; inner_cv_splits = 3
  group_kfold = GroupKFold(n_splits=outer_cv_splits)
  for train_indices, test_indices in group_kfold.split(X,y,users):
    out_X_train = X[train_indices]; out_y_train = y[train_indices]
    out_X_test = X[test_indices]; out_y_test = y[test_indices]

    # Inner CV
    strat_kfold = StratifiedKFold(n_splits=inner_cv_splits, random_state=0, shuffle=False)
    custom_cv_indices = []
    

  # Data partitioning

  # Generate candidate architectures
  models = modelgen.generate_models(X_train.shape, \
                                    number_of_classes=num_classes, \
                                    number_of_models=3)  

  # Compare generated architectures on a subset of data
  outfile = os.path.join(resultdir, 'model_comparison.json')
  hist, val_acc, val_loss = find_architecture.train_models_on_samples(X_train, \
                                 y_train, X_val, y_val, models, nr_epochs=5, \
                                 subset_size=300, verbose=True, \
                                 outputfile=outfile)

  # Choose best model and evaluate values on validation data
  best_model_index = np.argmax(val_acc)
  best_model, best_params, best_model_type = models[best_model_index]
  print('Best model type and parameters:')
  print(best_model_type)
  print(best_params)

  nr_epochs = 1
  history = best_model.fit(X_train, y_train, epochs=nr_epochs, \
                           validation_data=(X_val,y_val))
  
  # Save model
  best_model.save(os.path.join(resultdir,'best_model.h5'))

  # Predict probability on validation data
  probs = model.predict_proba(X_val, batch_size=1)
  y_pred = probs.argmax(axis=1)
  y_true = y_val.argmax(axis=1)
  conf_mat = pd.crosstab(pd.Series(y_true), pd.Series(y_pred)) 
  conf_mat.index = [sleep_states[idx] for idx in conf_mat.index]
  conf_mat.columns = [sleep_states[idx] for idx in conf_mat.columns]
  conf_mat.reindex(columns=[lbl for lbl in sleep_states], fill_value=0)
  print(conf_mat)

  

if __name__ == "__main__":
  main(sys.argv[1:])
