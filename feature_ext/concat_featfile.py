# -*- coding: utf-8 -*-
import sys
import pandas as pd

def main(argv):
    infile1 = argv[0]
    infile2 = argv[1]
    outfile = argv[2]
    
    df1 = pd.read_csv(infile1)
    df2 = pd.read_csv(infile2)
    out_df = pd.concat((df1,df2), axis=0)
    out_df.to_csv(outfile, sep=',', index=False)    
    
if __name__ == "__main__":
    main(sys.argv[1:])

