import sys,os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(argv):
  infile = argv[0]
  userfile = argv[1]
  outdir = argv[2]

  if not os.path.exists(os.path.join(outdir,'healthy')):
    os.makedirs(os.path.join(outdir,'healthy'))
  if not os.path.exists(os.path.join(outdir,'poor')):
    os.makedirs(os.path.join(outdir,'poor'))

  fname = os.path.basename(infile)

  user_df = pd.read_csv(userfile)
  healthy_sleepers = user_df[user_df['sleep_disorder'] == 0]['user']
  poor_sleepers = user_df[user_df['sleep_disorder'] == 1]['user']
 
  df = pd.read_csv(infile)

  # Healthy sleepers
  healthy_df = df[df['user'].isin(healthy_sleepers)].reset_index(drop=True)
  healthy_df.to_csv(os.path.join(outdir,'healthy',fname.replace('all','healthy')), index=False)
 
  # Poor sleepers
  poor_df = df[df['user'].isin(poor_sleepers)].reset_index(drop=True)
  poor_df.to_csv(os.path.join(outdir,'poor',fname.replace('all','poor')), index=False)

if __name__ == "__main__":
  main(sys.argv[1:])
