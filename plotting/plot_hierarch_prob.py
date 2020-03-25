import sys,os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

def main(argv):
  infile = argv[0]
  outdir = argv[1]

  df = pd.read_csv(infile)
  states = [col.split('true_')[1] for col in df.columns if col.startswith('true')]

  bins = 50
  for state in states:
    state_true = df['true_'+state].values
    state_pred = df['smooth_'+state].values
    st_hist, st_edges = np.histogram(state_pred[state_true == 1], bins=bins)
    st_hist = st_hist / float(state_true.sum())
    st_bins = (st_edges[:-1] + st_edges[1:])/2.0
    plt.plot(st_bins, st_hist, linewidth=2)
    plt.xlabel([0,1])
    plt.ylabel([0,1])
    plt.xlabel('Predicted probability')
    plt.ylabel('Normalized frequency')
    plt.title(state, fontsize=30)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, state+'_hierarch_pred_prob.jpg'))
    plt.close()

if __name__ == "__main__":
  main(sys.argv[1:])
