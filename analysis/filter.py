import sys,os
import numpy as np
import h5py
from scipy import signal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Save time series plot
def save_ts_plot(x,outfile,y=None,z=None):     
    strip_samp = int(len(x)/10)
    nsamp = len(x)
    nstrips = nsamp // strip_samp
    
    plt.figure(num=1, figsize=(2*nstrips,3*nstrips))
    fig, axes = plt.subplots(nstrips,1,sharex=True,num=1)
    for strip in range(nstrips):
        x_strip = x[strip*strip_samp:(strip+1)*strip_samp]
        xaxis = range(len(x_strip))
        axes[strip].plot(xaxis, x_strip, color='red')
        if y is not None:
            y_strip = y[strip*strip_samp:(strip+1)*strip_samp]
            axes[strip].plot(xaxis, y_strip, color='green')
        if z is not None:
            z_strip = z[strip*strip_samp:(strip+1)*strip_samp]
            axes[strip].plot(xaxis, z_strip, color='blue')
        #axes[strip].set_ylim(min_ylim, max_ylim)
        
    plt.savefig(outfile)
    plt.close('all')
    
def main(argv):
    infile = argv[0]
    samp_freq = float(argv[1])
    cutoff_freq = float(argv[2])
    outdir = argv[3]

    fh = h5py.File(infile, 'r')

    # Extract data info
    x = np.array(fh['X'])
    y = np.array(fh['Y'])
    z = np.array(fh['Z'])
    
    # Plot unfiltered data 
    save_ts_plot(x,os.path.join(outdir,'xyz_before_filtering.jpg'),y,z)

    # Plot EN before filtering
    EN = np.sqrt(x*x + y*y + z*z)
    save_ts_plot(EN,os.path.join(outdir,'EN_before_filtering.jpg'))

    # Apply Butterworth filter
    Wn = cutoff_freq / (samp_freq/2.0)
    b,a = signal.butter(N=4, Wn=Wn, btype='lowpass')
    x_filt = signal.filtfilt(b, a, x)
    y_filt = signal.filtfilt(b, a, y)
    z_filt = signal.filtfilt(b, a, z)
    EN_filt = signal.filtfilt(b, a, EN)
    
    # Plot filtered data 
    save_ts_plot(x_filt,os.path.join(outdir,'xyz_after_filtering_'+str(Wn)+'.jpg'),y_filt,z_filt)

    # Plot EN after filtering
    #EN = np.sqrt(x*x + y*y + z*z)
    save_ts_plot(EN_filt,os.path.join(outdir,'EN_after_filtering_'+str(Wn)+'.jpg'))

if __name__ == "__main__":
  main(sys.argv[1:])
