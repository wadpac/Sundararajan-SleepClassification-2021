import sys,os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def main(argv):
  infile = argv[0]
  outfile = argv[1]

  df = pd.read_csv(infile)

  labels = list(df['label'].values)
  sleep_states = ['Wake','NREM 1','NREM 2','NREM 3','REM']
  ctr = Counter(labels)
  ctr = {key:val for key,val in ctr.items() if key in sleep_states}
  total = sum(ctr.values())
  ctr = {key:val*100.0/total for key,val in ctr.items()}
  
  plt_lbl = sorted(ctr.keys())
  plt_values = [ctr[key] for key in plt_lbl]
  plt.Figure()
  plt.pie(plt_values, labels=plt_lbl, autopct='%1.0f%%', pctdistance=0.6, 
          labeldistance=1.1, textprops={'fontsize': 14})
          #colors=['red','blue','green','magenta','cyan'])
  plt.tight_layout()
  plt.savefig(outfile)

if __name__ == "__main__":
  main(sys.argv[1:])
