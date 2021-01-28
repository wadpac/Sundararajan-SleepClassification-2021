
# Specify file paths:

# The models you downloaded from https://doi.org/10.5281/zenodo.3752645
path2models = 'C:/Research/actigraphy/data/results/all/models'

# The R script you can download here: https://github.com/wadpac/SleepStageClassification/blob/master/ggir_ext/get_sleep_stage.R
path2get_sleep_stageR = "C:/Research/actigraphy/code/SleepStageClassification/ggir_ext/get_sleep_stage.R"

# Directory where the raw accelation files are
datadir = "C:/Research/actigraphy/data/psgnewcastle2015/test_acc/"

# Directory where you want GGIR to store the results
outputdir = "C:/Research/actigraphy/code/SleepStageClassification/ggir_ext/myresults"

# The next 3 lines are specific to your local setup, see reticulate documentation
# for specifying Python environment
library("reticulate")
path2condaenv = "C:/Users/KalaivaniSundararaja/Anaconda3/envs/GGIR/"
use_python(path2condaenv, required=TRUE)
sys <- import("sys", convert = FALSE)
sys$path$append(".") # path for features.py and utils.py

#===============================================================
# Mode
getmode <- function(v) {
  uniqv <- unique(v)
  mode <- uniqv[which.max(tabulate(match(v, uniqv)))]
  return(mode)
}

source(path2get_sleep_stageR)
myfun = list(FUN=get_sleep_stage,
             parameters= c(30,path2models,'binary'), #time interval for feature aggregation, path to model directory, mode (binary, nonwear, multiclass)
             expected_sample_rate= 30,
             expected_unit="g",
             colnames = c("wake_sleep"),
             minlength = 5,
             outputres = 30,
             outputtype="character",
             aggfunction = getmode,
             timestamp=T)
library("GGIR")
g.shell.GGIR(datadir=datadir,
             outputdir=outputdir,
             mode=1:2,
             epochvalues2csv = TRUE,
             do.report=2,
             myfun=myfun,
             do.parallel=F)