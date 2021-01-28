# Sleep classification with GGIR

This repository consists of code to integrate Nonwear and Wake-Sleep classification developed using Python with [GGIR](https://cran.r-project.org/web/packages/GGIR/index.html). Given a set of raw accelerometer files, this code yields Wake/Sleep/Nonwear classification for every 30 seconds of data. 

## Machine learned models
The Random Forests models for Wake-Sleep and Nonwear classification can be found [here](https://doi.org/10.5281/zenodo.3752645
). Note that this is a collection of Random forest models which are applied jointly as an ensemble classifier.


## To apply this model to your data do:
1. Download and unzip the machine learned models: https://doi.org/10.5281/zenodo.3752645
2. Clone or copy all files from https://github.com/wadpac/SleepStageClassification/blob/master/ggir_ext
3. Update the file paths at the top of `main.R`
4. Update the path to `get_sleep_stage.py` inside `get_sleep_stage.R`
5. Install GGIR, e.g. with `install.packages("GGIR", dependencies=TRUE)`
5. Source `main.R`


**NOTE:**
These models were trained with scikit-learn version 0.22.1. Loading these models with other scikit-learn versions can cause incompatibility errors/warnings.
