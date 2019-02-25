# SleepStageClassification
Classification of sleep stages using accelerometer data

## Preprocessing
### 1. Get raw data
Use get_raw_data_psgnewcastle.R or get_raw_data_UPenn.R to extract raw data (X,Y,Z,battery/button,light,temperature and timestamps) from .bin and .cwa files respectively. The extracted data is stored in HDF5 format.

### 2. Get calibration parameters and nonwear information
Use get_calib_nonwear.R to obtain calibration parameters and nonwear epochs using GGIR and store them in CSV files

### 3. Create preprocessed dataset by applying autocalibration and aligning nonwear and label information
Use preproc_psgnewcastle.py and preproc_UPenn.py to preprocess .bin and .cwa files respectively using the data extracted in Steps 1 and 2. Input parameters include directory path to extracted raw data, path to calibration parameters, path to nonwear information, path to label data and output path. This step applies calibration parameters to the extracted raw data and aligns nonwear and label information to the extracted data based on overlapping timestamps.
