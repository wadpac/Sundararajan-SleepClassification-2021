import sys,os
import h5py
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def save_ts_plot(x, y, z, label, nonwear, states, outfile):     
    nsamp = len(x)
    nstrips = 10
    strip_samp = nsamp // nstrips
    
    plt.figure(num=1, figsize=(2*nstrips,3*nstrips))
    fig, axes = plt.subplots(nstrips,1,sharex=True,num=1)
    min_lim = min(np.minimum(x,y,z))
    max_lim = max(np.maximum(x,y,z))
    for strip in range(nstrips):
        x_strip = x[strip*strip_samp:(strip+1)*strip_samp]
        y_strip = y[strip*strip_samp:(strip+1)*strip_samp]
        z_strip = z[strip*strip_samp:(strip+1)*strip_samp]
        lbl = label[strip*strip_samp:(strip+1)*strip_samp]
        nw = nonwear[strip*strip_samp:(strip+1)*strip_samp]
        t = range(len(x_strip))
        axes[strip].plot(t,x_strip, color='red')
        axes[strip].plot(t,y_strip, color='green')
        axes[strip].plot(t,z_strip, color='blue')
        axes[strip].set_ylim(min_lim, max_lim)
        st1 = axes[strip].fill_between(t, min_lim, max_lim, where=(lbl==states[0]), facecolor='green', alpha=0.3)
        st2 = axes[strip].fill_between(t, min_lim, max_lim, where=(lbl==states[1]), facecolor='blue', alpha=0.3)
        st3 = axes[strip].fill_between(t, min_lim, max_lim, where=(lbl==states[2]), facecolor='yellow', alpha=0.3)
        st4 = axes[strip].fill_between(t, min_lim, max_lim, where=(lbl==states[3]), facecolor='magenta', alpha=0.3)
        st5 = axes[strip].fill_between(t, min_lim, max_lim, where=(lbl==states[4]), facecolor='cyan', alpha=0.3)
        axes[strip].fill_between(t, min_lim, max_lim, where= nw==True, facecolor='red', alpha=0.5)
    fig.legend((st1,st2,st3,st4,st5), states, loc='lower center', ncol=5)
        
    plt.savefig(outfile)
    plt.close('all')
    
def main(argv):
    infile = argv[0]
    outfile = argv[1]
  
    states = ['Wake','NREM 1','NREM 2','NREM 3','REM']
    
    fh = h5py.File(infile, 'r')
    x = np.array(fh['X'])
    y = np.array(fh['Y'])
    z = np.array(fh['Z'])
    timestamp = pd.Series(fh['DateTime']).apply(lambda x: x.decode('utf8'))
    timestamp = pd.to_datetime(timestamp, format='%Y-%m-%d %H:%M:%S.%f')
    
    # Get nonwear for each interval
    nonwear = np.array(fh['Nonwear'])
        
    # Standardize label names for both datasets
    # Get label for each interval
    label = np.array([lbl.decode('utf8') for lbl in np.array(fh['SleepState'])], dtype=object)
    label[label == 'W'] = 'Wake'
    label[label == 'N1'] = 'NREM 1'
    label[label == 'N2'] = 'NREM 2'
    label[label == 'N3'] = 'NREM 3'
    label[label == 'R'] = 'REM'
          
    save_ts_plot(x, y, z, label, nonwear, states, outfile)

if __name__ == '__main__':
    main(sys.argv[1:])
