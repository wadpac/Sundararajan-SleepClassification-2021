# -*- coding: utf-8 -*-
import sys
import pandas as pd

def main(argv):
  infile1 = argv[0]
  infile2 = argv[1]
  infile3 = argv[2]
  outfile = argv[3]
  
  df1 = pd.read_csv(infile1)
  df2 = pd.read_csv(infile2)
  df3 = pd.read_csv(infile3)
  out_df = pd.concat((df1,df2), axis=0)
  out_df = pd.concat((out_df,df3), axis=0)
  out_df.to_csv(outfile, sep=',', index=False)    
    
if __name__ == "__main__":
    main(sys.argv[1:])

