import sys,os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from networkx import DiGraph
from networkx import relabel_nodes
from sklearn_hierarchical_classification.constants import ROOT
from tqdm import tqdm
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

def recursive_predict(graph, classes, class_prob, node):
  # If node is leaf, return it
  if len(list(graph.successors(node))) == 0:
      return node
  indices = [classes.index(child_node_id) for child_node_id in graph.successors(node)]
  probs = class_prob[indices]
  pred_idx = np.argmax(probs)
  pred_node = classes[indices[pred_idx]]
  pred = recursive_predict(graph, classes, class_prob, pred_node)
  return pred     

def get_multilabel(pred, graph):
  leaf_nodes = [node for node in graph.nodes() if (graph.out_degree(node) == 0)\
                                                   and (graph.in_degree(node) == 1)]
  nodes = [node for node in graph.nodes() if node != '<ROOT>']
  multilabel_pred = np.zeros((pred.shape[0], len(nodes)))
  for i in range(pred.shape[0]):
    node = pred[i]
    while (node != '<ROOT>'):
      multilabel_pred[i,node] = 1
      predecessors = [idx for idx in graph.predecessors(node)]
      node = predecessors[0] # only one parent per node
  return multilabel_pred

def get_confusion_matrix(y_true, y_pred, states, plot_states):
  y_true_relabel = np.zeros(y_true.shape)
  y_pred_relabel = np.zeros(y_pred.shape)
  for new_idx,lbl in enumerate(plot_states):
    old_idx = states.index(lbl)
    y_true_relabel[:,new_idx] = y_true[:,old_idx]
    y_pred_relabel[:,new_idx] = y_pred[:,old_idx]
  conf_mat = np.dot(y_true_relabel.T, y_pred_relabel)
  denom = y_true_relabel.sum(axis=0)
  conf_mat = conf_mat.astype(float) / denom.reshape(-1,1)
  return conf_mat

def main(argv):
  infile = argv[0]
  mode = argv[1]
  dataset = argv[2]
  outdir = argv[3]

  # Class hierarchy for sleep stages
  class_hierarchy = {
    ROOT : {"Wear", "Nonwear"},
    "Wear" : {"Wake", "Sleep"},
    "Sleep" : {"NREM", "REM"},
    "NREM" : {"Light", "NREM 3"},
    "Light" : {"NREM 1", "NREM 2"} 
  }
  
  graph = DiGraph(class_hierarchy)    
 
  df = pd.read_csv(infile)
  nfolds = len(set(df['Fold']))
  sleep_states = [col.split('_')[1] for col in df.columns if col.startswith('true')]
  sleep_labels = [idx for idx,state in enumerate(sleep_states)]
  true_cols = [col for col in df.columns if col.startswith('true')]
  pred_cols = [col for col in df.columns if col.startswith('smooth')]
  nclasses = len(true_cols)
  
  node_label_mapping = {
      old_label: new_label
      for new_label, old_label in enumerate(list(sleep_states))
  }
  graph = relabel_nodes(graph, node_label_mapping)

  plot_states = ['Nonwear', 'Wear', 'Wake', 'Sleep', 'NREM', 'REM',\
                 'Light', 'NREM 3', 'NREM 1', 'NREM 2']
  confusion_mat = np.zeros((len(sleep_states),len(sleep_states)))
  for fold in range(nfolds):
    true_prob = df[df['Fold'] == fold+1][true_cols].values  
    pred_prob = df[df['Fold'] == fold+1][pred_cols].values 
    y_pred = []
    for i in tqdm(range(pred_prob.shape[0])):
      pred = recursive_predict(graph, list(range(len(sleep_states))), pred_prob[i], '<ROOT>')
      y_pred.append(pred)
    y_pred = np.array(y_pred)
    y_pred = get_multilabel(y_pred, graph).astype(int)
    fold_conf_mat = get_confusion_matrix(true_prob, y_pred, sleep_states, plot_states)
    confusion_mat = confusion_mat + fold_conf_mat
  confusion_mat = confusion_mat*100.0 / nfolds

  # Plot confusion matrix
  plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
  plt.colorbar()
  tick_marks = np.arange(len(sleep_states))
  plt.xticks(tick_marks, plot_states, rotation=45)
  plt.yticks(tick_marks, plot_states)

  thresh = confusion_mat.max() / 2.0
  for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1])):
    plt.text(j, i, '{:0.2f}'.format(confusion_mat[i, j]),\
             horizontalalignment="center", fontsize=9,\
             color="white" if confusion_mat[i, j] > thresh else "black")

  plt.ylabel('True label', fontsize=14)
  plt.xlabel('Predicted label', fontsize=14)
  plt.tight_layout()
  plt.savefig(os.path.join(outdir, '-'.join((dataset, mode, 'confmat')) + '.jpg'))

if __name__ == "__main__":
  main(sys.argv[1:])
