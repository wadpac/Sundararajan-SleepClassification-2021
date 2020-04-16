get_sleep_stage = function(data=c(), parameters=c()) {
  # data: 4 column matrix with timestamp and acc data
  # parameters: time interval for feature computation, model directory, mode
  # mode: binary, multiclass, nonwear
  time_interval = parameters[1]
  modeldir = parameters[2]
  mode = parameters[3]
  
  timestamp <- as.character(as.POSIXct(data[,1], origin='1970-01-01', tz='UTC'))

  reticulate::source_python("C:/Research/actigraphy/code/SleepStageClassification/ggir_ext/get_sleep_stage.py")
  result = get_sleep_stage(data, time_interval, modeldir, mode)
  df = as.data.frame(matrix(c(result)))
  return(df)
}