library(GENEAread)
library(rhdf5)

indir <- 'C:/Research/actigraphy/data/amc_new/amc_extracted/wrap/'
outdir <- 'C:/Research/actigraphy/data/amc_new/raw_data/'
dir.create(outdir, showWarnings = FALSE)

files <- list.files(path = indir, pattern='*.bin')
lapply(files, function(fname) {
  out_fname <- paste(substr(fname,1,nchar(fname)-4),'.h5',sep='')
  if (file.exists(file.path(outdir,out_fname))){
    print(paste('Finished processing ',fname))
    return(NULL)
  }
  accdata <- read.bin(file.path(indir,fname), calibrate=TRUE)
  timestamp <- as.character(as.POSIXct(accdata$data.out[,1], origin='1970-01-01', tz='UTC'))
  X <- accdata$data.out[,2]
  Y <- accdata$data.out[,3]
  Z <- accdata$data.out[,4]
  light <- accdata$data.out[,5]
  button <- accdata$data.out[,6]
  temp <- accdata$data.out[,7]
  
  out_fname <- paste(substr(fname,1,nchar(fname)-4),'.h5',sep='')
  h5write(timestamp, file=file.path(outdir,out_fname), 'timestamp')
  h5write(X, file=file.path(outdir,out_fname), 'X')
  h5write(Y, file=file.path(outdir,out_fname), 'Y')
  h5write(Z, file=file.path(outdir,out_fname), 'Z')
  h5write(temp, file=file.path(outdir,out_fname), 'temp')
  h5write(button, file=file.path(outdir,out_fname), 'button')
  h5write(light, file=file.path(outdir,out_fname), 'light')
  print(paste('Finished processing ',fname))
})