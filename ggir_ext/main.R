rm(list=ls())
# Mode
getmode <- function(v) {
  uniqv <- unique(v)
  mode <- uniqv[which.max(tabulate(match(v, uniqv)))]
  return(mode)
}
get_sleep_stage = "/home/vincent/projects/SleepStageClassification/ggir_ext/get_sleep_stage.R" #"C:/Research/actigraphy/code/ggir_ext/get_sleep_stage.R"
model_location = "/media/vincent/DATA/random_forests_kalaivani" #'C:/Research/actigraphy/data/results/all/models'
loc_get_sleep_stage_py = "/home/vincent/projects/SleepStageClassification/ggir_ext/get_sleep_stage.py"
datadir = "/media/vincent/DATA/whitehallNapDetection"

path_to_ggir_ext = "/home/vincent/projects/SleepStageClassification/ggir_ext"

outputdir = "/media/vincent/projects/Kalaivani"

library("reticulate")
# use_python("C:/Users/KalaivaniSundararaja/Anaconda3/envs/GGIR/", required=TRUE)

dirR = "/home/vincent/GGIR/R"
ffnames = dir(dirR) # creating list of filenames of scriptfiles to load
for (i in 1:length(ffnames))  source(paste(dirR,"/",ffnames[i],sep="")) #loading scripts for reading geneactiv data
library("Rcpp")
pathR = "/home/vincent/GGIR"
sourceCpp(paste0(pathR,"/src/numUnpack.cpp"))
sourceCpp(paste0(pathR,"/src/resample.cpp"))

library("reticulate")
use_virtualenv("~/projects/venv_GGIR", required = TRUE) # Local Python environment
# py_install("pandas", pip = TRUE)
# py_install("numpy", pip = TRUE)
# py_install("pickle-mixin", pip = TRUE)
# py_install("scipy", pip = TRUE)
# py_install("sklearn", pip = TRUE)

sys <- import("sys", convert = FALSE)
sys$path$append(path_to_ggir_ext) # path for features.py and utils.py

source(get_sleep_stage)
myfun = list(FUN=get_sleep_stage,
             parameters= c(30, model_location,'binary', 
                           loc_get_sleep_stage_py), #time interval for feature aggregation, path to model directory, mode (binary, nonwear, multiclass)
             expected_sample_rate= 30,
             expected_unit="g",
             colnames = c("wake_sleep"),
             minlength = 5,
             outputres = 30,
             outputtype="character",
             aggfunction = getmode,
             timestamp=T,
             reporttype = "type")
library("GGIR")
g.shell.GGIR(datadir=datadir,
             outputdir=outputdir,
             mode=1,
             print.filename=TRUE,
             chunksize= 0.7,
             do.cal=T,
             do.visual=T,
             f0=2,f1=10,
             outliers.only=F,
             epochvalues2csv = TRUE,
             do.report=c(),
             overwrite=F,
             myfun=myfun,
             do.parallel=F)