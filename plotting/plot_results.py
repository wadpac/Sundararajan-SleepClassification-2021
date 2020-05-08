import sys,os
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style='whitegrid')
import matplotlib.pyplot as plt

def main(argv):
  outfile = argv[0]
  metric = ['Precision','Precision','Precision','Recall','Recall','Recall','F-score','F-score','F-score']
  method = ['Heuristic','Random Forests','Deep learning','Heuristic','Random Forests','Deep learning','Heuristic','Random Forests','Deep learning']
  values = [69.23, 71.10, 72.18, 68.68, 74.53, 64.41, 68.94, 72.57, 66.96]
  df = pd.DataFrame({'metric': metric, 'method': method, 'values': values})
  ax = sns.barplot(x='metric', y='values', hue='method', data=df)
  ax.set(xlabel='Performance metric', ylabel='Values (%)')
  plt.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
  plt.title('Wake vs. Sleep classification on AMC dataset (105 subjects)')
  plt.ylim([0,100])
  
  def plot_val(metric, st_idx):
    ax.text(metric-0.4,values[st_idx]+2,str(values[st_idx]))
    ax.text(metric-0.1,values[st_idx+1]+2,str(values[st_idx+1]))
    ax.text(metric+0.2,values[st_idx+2]+2,str(values[st_idx+2]))
  plot_val(0,0)
  plot_val(1,3)
  plot_val(2,6)

  fig = ax.get_figure()
  fig.savefig(outfile, bbox_inches='tight')

if __name__ == '__main__':
  main(sys.argv[1:])
