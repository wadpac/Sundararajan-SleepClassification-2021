# Sleep classification with GGIR

This repository consists of code to integrate Nonwear and Wake-Sleep classification developed using Python with [GGIR](https://cran.r-project.org/web/packages/GGIR/index.html). Given a set of raw accelerometer files, this code yields Wake/Sleep/Nonwear classification for every 30 seconds of data. 

## Machine learning models
The Random Forests models for Wake-Sleep and Nonwear classification can be found [here](https://doi.org/10.5281/zenodo.3752645
). 
** NOTE: **
These models were trained with scikit-learn version 0.22.1. Loading these models with other scikit-learn versions can cause incompatibility errors/warnings.
