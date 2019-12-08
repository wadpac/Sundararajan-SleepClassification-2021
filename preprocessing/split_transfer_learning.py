import sys,os
import random
import pandas as pd
import numpy as np

def main(argv):
  infile = argv[0]
  outdir = argv[1]
  
  df = pd.read_csv(infile)
  users = list(set(df['user']))
  random.shuffle(users)
  
  val_len = int(0.3*len(users))
  val_users = users[:val_len]
  val_df = df[df['user'].isin(val_users)]
  val_df.to_csv(os.path.join(outdir, 'val_'+os.path.basename(infile)), index=False)
  
  test_users = users[val_len:]
  test_df = df[df['user'].isin(test_users)]
  test_df.to_csv(os.path.join(outdir, 'test_'+os.path.basename(infile)), index=False)

if __name__ == "__main__":
  main(sys.argv[1:])
