import sys,os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def save_agg_plot(df, col_str, order, outfile):
    plt.figure()
    df[col_str] = df[col_str].astype(float)
    sns.boxplot(x="label", y=col_str, data=df, order=order)
    plt.savefig(outfile)
    plt.close('all')
    
def main(argv):
    infile = argv[0]
    outfile = argv[1]

    df = pd.read_csv(infile, dtype={'label':object, 'user':object, 'position':object, 'dataset':object})
    orig_cols = df.columns
    sleep_states = ['Wake','NREM 1','NREM 2','NREM 3','REM']
    df = df[df['label'].isin(sleep_states)].reset_index()
    df = df[orig_cols]
    
    save_agg_plot(df, 'angz_mean', sleep_states, outfile)

if __name__ == "__main__":
    main(sys.argv[1:])
