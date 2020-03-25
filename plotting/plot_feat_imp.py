import sys,os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})

def main(argv):
  infile = argv[0]
  mode = argv[1]
  dataset = argv[2]
  outdir = argv[3]

  df = pd.read_csv(infile)
  fold_cols = [col for col in df.columns if col.startswith('Fold')]
  df['importance_mean'] = df[fold_cols].mean(axis=1)
  df['importance_std'] = df[fold_cols].std(axis=1)
  df = df.sort_values(by='importance_mean', ascending=True)
 
  plt.Figure()
  plt.barh(df['Features'], width=df['importance_mean'], xerr=df['importance_std'])
  plt.xlabel('Feature importance', fontsize=14)
  plt.tight_layout()
  plt.savefig(os.path.join(outdir, '-'.join(('featimp',mode,dataset))+'.jpg'))
  plt.close()

if __name__ == "__main__":
  main(sys.argv[1:])
