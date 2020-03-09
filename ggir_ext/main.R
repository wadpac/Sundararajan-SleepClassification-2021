# Mode
getmode <- function(v) {
  uniqv <- unique(v)
  mode <- uniqv[which.max(tabulate(match(v, uniqv)))]
  return(mode)
}

library("reticulate")
use_python("C:/Users/KalaivaniSundararaja/Anaconda3/envs/GGIR/", required=TRUE)
sys <- import("sys", convert = FALSE)
sys$path$append(".") # path for features.py and utils.py

source("C:/Research/actigraphy/code/ggir_ext/get_sleep_stage.R")
myfun = list(FUN=get_sleep_stage,
             parameters= c(30,'C:/Research/actigraphy/data/results/all/models','binary'), #time interval for feature aggregation, path to model directory, mode (binary, nonwear, multiclass)
             expected_sample_rate= 30,
             expected_unit="g",
             colnames = c("wake_sleep"),
             minlength = 5,
             outputres = 30,
             outputtype="character",
             aggfunction = getmode,
             timestamp=T)
library("GGIR")
g.shell.GGIR(datadir="C:/Research/actigraphy/data/psgnewcastle2015/test_acc/",
             outputdir="C:/Research/actigraphy/code/ggir_ext/myresults",
             mode=1:2,
             epochvalues2csv = TRUE,
             do.report=2,
             myfun=myfun,
             do.parallel=F)