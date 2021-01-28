# Sundararajan Sleep-Wake-Nonwear classification via GGIR

This repository consists of code to integrate the Nonwear and Wake-Sleep classification as presented in [Sundararajan 2021](https://www.nature.com/articles/s41598-020-79217-x) by combining the Python functions shared in this repository with R package [GGIR](https://cran.r-project.org/web/packages/GGIR/index.html). Given a set of raw accelerometer files, this code yields Wake/Sleep/Nonwear classification for every 30 seconds of data. The Random Forests models for Wake-Sleep and Nonwear classification can be found [here](https://doi.org/10.5281/zenodo.3752645
). Note that this is a collection of Random forest models which are applied jointly as an ensemble classifier.


## To apply this model to your data do:
1. Download and unzip the machine learned models: https://doi.org/10.5281/zenodo.3752645
2. Clone or copy all files from https://github.com/wadpac/SleepStageClassification/blob/master/ggir_ext
3. Update the file paths at the top of `main.R`
4. Update the path to `get_sleep_stage.py` inside `get_sleep_stage.R`
5. Install GGIR, e.g. with `install.packages("GGIR", dependencies=TRUE)`
6. Source `main.R` and GGIR will start processing your data files. See GGIR [documentation](https://cran.r-project.org/web/packages/GGIR/vignettes/GGIR.html) for extra clarifation on the output.

**NOTE 1:**
These models were trained with scikit-learn version 0.22.1. Loading these models with other scikit-learn versions can cause incompatibility errors/warnings.

**NOTE 2:**
As you will notice the R code uses R package reticulate as interface to Python. At the moment this construction does not facilitate parallel processing of multiple data files like how GGIR is able to do when it uses its own vanHees heuristic.


# Sundararajan Sleep-Wake-Nonwear classification in Python (without R interface)

Use https://github.com/wadpac/SleepStageClassification/blob/master/ggir_ext/get_sleep_stage.py
