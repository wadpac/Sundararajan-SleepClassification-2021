# Ref: https://nbviewer.jupyter.org/github/terryum/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/Example_DataAugmentation_TimeseriesData.ipynb
# T. T. Um et al., “Data augmentation of wearable sensor data for parkinson’s disease monitoring 
# using convolutional neural networks,” in Proceedings of the 19th ACM International Conference 
# on Multimodal Interaction, ser. ICMI 2017. New York, NY, USA: ACM, 2017, pp. 216–220.
import math
import numpy as np
from scipy.interpolate import CubicSpline      # for warping

# X is a LxMxN timeseries signal with L samples, M timesteps and N channels (easier batch processing for L samples)

# Jitter - adds additive noise
def jitter(X, sigma=0.05):
  noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
  return X + noise

# Scaling - changes the magnitude of the data in a window by multiplying by a random scalar 
def scaling(X, sigma=0.1):
  sc_factor = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0],1,X.shape[2]))
  noise = np.matmul(np.ones((X.shape[0],X.shape[1],1)), sc_factor)
  return X*noise
  
# Magnitude warping - changes the magnitude of each sample by convolving 
# the data window with a smooth curve varying around one 
def generate_random_curves(X, sigma=0.2, knot=4):
  random_curves = np.zeros(X.shape)
  for i in range(X.shape[0]):
    xx = (np.ones((X.shape[2],1))*(np.arange(0,X.shape[1], (X.shape[1]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[2]))
    x_range = np.arange(X.shape[1])
    cs_x = CubicSpline(xx[:,0], yy[:,0])
    cs_y = CubicSpline(xx[:,1], yy[:,1])
    cs_z = CubicSpline(xx[:,2], yy[:,2])
    random_curves[i,:,:] = np.array([cs_x(x_range),cs_y(x_range),cs_z(x_range)]).transpose()
  return random_curves

def magnitude_warp(X, sigma=0.2):
    return X * generate_random_curves(X, sigma)

# Time warping - by smoothly distorting the time intervals between samples, 
# the temporal locations of the samples are changed 
def distort_timesteps(X, sigma=0.2):
  tt = generate_random_curves(X, sigma) # Regard these samples around 1 as time intervals
  tt_cum = np.cumsum(tt, axis=1)        # Add intervals to make a cumulative graph
  # Make the last value to have X.shape[1]
  t_scale = np.array([(X.shape[1]-1)/tt_cum[:,-1,0],(X.shape[1]-1)/tt_cum[:,-1,1],(X.shape[1]-1)/tt_cum[:,-1,2]]).transpose()
  t_scale = np.matmul(np.ones((X.shape[0],X.shape[1],1)),t_scale.reshape(X.shape[0],1,X.shape[2]))
  tt_cum = tt_cum*t_scale
  return tt_cum

def time_warp(X, sigma=0.2):
  tt_new = distort_timesteps(X, sigma)
  X_new = np.zeros(X.shape)
  for i in range(X.shape[0]):
    X_new[i,:,0] = np.interp(np.arange(X.shape[1]), tt_new[i,:,0], X[i,:,0])
    X_new[i,:,1] = np.interp(np.arange(X.shape[1]), tt_new[i,:,1], X[i,:,1])
    X_new[i,:,2] = np.interp(np.arange(X.shape[1]), tt_new[i,:,2], X[i,:,2])
  return X_new

# Rotation - applying arbitrary rotations to the existing data can be used as
# a way of simulating different sensor placements
def get_rotation_matrices(angle, axis='x'):  
  cos_angle = np.cos(angle)
  sin_angle = np.sin(angle)

  rot_mat = np.zeros((angle.shape[0],3,3))
  if axis == 'x':
     rot_mat[:,0,0] = 1
     rot_mat[:,1,1] = cos_angle[:,0]
     rot_mat[:,1,2] = -sin_angle[:,0]
     rot_mat[:,2,1] = sin_angle[:,0]
     rot_mat[:,2,2] = cos_angle[:,0]
  elif axis == 'y':
     rot_mat[:,0,0] = cos_angle[:,0]
     rot_mat[:,0,2] = -sin_angle[:,0]
     rot_mat[:,1,1] = 1
     rot_mat[:,2,0] = sin_angle[:,0]
     rot_mat[:,2,2] = cos_angle[:,0]
  elif axis == 'z':
     rot_mat[:,0,0] = cos_angle[:,0]
     rot_mat[:,0,1] = -sin_angle[:,0]
     rot_mat[:,1,0] = sin_angle[:,0]
     rot_mat[:,1,1] = cos_angle[:,0]
     rot_mat[:,2,2] = 1

  return rot_mat

def rotation(X):
  ang_lim = 5.0*np.pi/180.0 # maximum angle variation is +/- 5 degrees
  angle = np.random.uniform(low=-ang_lim, high=ang_lim, size=(X.shape[0],X.shape[2]))
  
  Rx = get_rotation_matrices(angle, axis='x')
  Ry = get_rotation_matrices(angle, axis='y')
  Rz = get_rotation_matrices(angle, axis='z')
  
  R = np.matmul(Rz, np.matmul(Ry,Rx))
                                          
  return np.matmul(X , R)

# DO NOT USE Permutation for accelerometry data since it may mess up the sequential properties, frequency spectrum and transitions
## Permutation - randomly perturb the temporal location of within-window events. 
## To perturb the location of the data in a single window, we first slice the data 
## into N samelength segments, with N ranging from 1 to 5, and randomly permute 
## the segments to create a new window
#def permutation(X, nPerm=4, minSegLength=10):
#  X_new = np.zeros(X.shape)
#  idx = np.random.permutation(nPerm)
#  bWhile = True
#  while bWhile == True:
#    segs = np.zeros(nPerm+1, dtype=int)
#    segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
#    segs[-1] = X.shape[0]
#    if np.min(segs[1:]-segs[0:-1]) > minSegLength:
#      bWhile = False
#  pp = 0
#  for ii in range(nPerm):
#    x_temp = X[segs[idx[ii]]:segs[idx[ii]+1],:]
#    X_new[pp:pp+len(x_temp),:] = x_temp
#    pp += len(x_temp)
#  return(X_new)

# Random sampling - randomly sample timesteps and interpolate data inbetween - same timesteps for all three axes
def rand_sample_timesteps(X, nSample=1000):
  tt = np.zeros((X.shape[0],nSample), dtype=int)
  tt[:,1:-1] = np.sort(np.random.randint(1,X.shape[1]-1,(X.shape[0],nSample-2)),axis=1)
  tt[:,-1] = X.shape[1]-1
  return tt

def rand_sampling(X, low=0.6, high=0.8):
  nSample = np.random.randint(int(low*X.shape[1]),int(high*X.shape[1]),1)[0]
  tt = rand_sample_timesteps(X, nSample)
  X_new = np.zeros(X.shape)
  for i in range(X.shape[0]):
    X_new[i,:,0] = np.interp(np.arange(X.shape[1]), tt[i], X[i,tt[i],0])
    X_new[i,:,1] = np.interp(np.arange(X.shape[1]), tt[i], X[i,tt[i],1])
    X_new[i,:,2] = np.interp(np.arange(X.shape[1]), tt[i], X[i,tt[i],2])
  return X_new

# Get Euclidean Norm minus One
def get_ENMO(x,y,z):
  enorm = np.sqrt(x*x + y*y + z*z)
  ENMO = np.maximum(enorm-1.0, 0.0)
  return ENMO

# Get tilt angles
def get_angle_z(x,y,z):
  angle_z = np.arctan2(z, np.sqrt(x*x + y*y)) * 180.0/math.pi
  return angle_z

# Get Locomotor Inactivity During Sleep - use convolve instead of rolling sum over intervals 
def get_LIDS(x,y,z):
  enmo = get_ENMO(x,y,z)
  enmo_sub = np.where(enmo < 0.02, 0, enmo-0.02) # assuming ENMO is in g
  win_sz = 21 # use smaller window size instead of 10-min rolling sum
  enmo_sub_smooth = np.apply_along_axis(lambda row: np.convolve(row, np.ones((win_sz,)), 'same'), axis=-1, arr=enmo_sub)
  lids = 100.0 / (enmo_sub_smooth + 1.0)
  win_sz = 71 # use larger window size instead of 30-min rolling average
  lids_smooth = np.apply_along_axis(lambda row: np.convolve(row, np.ones((win_sz,)), 'same'), axis=-1, arr=lids)/float(win_sz)
  return lids_smooth
