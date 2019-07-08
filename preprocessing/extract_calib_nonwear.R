library(GGIR)

indir <- 'C:/Research/actigraphy/data/amc_new/output_wrap/meta/basic/'
calibdir <- 'C:/Research/actigraphy/data/amc_new/calib_param/'
nonweardir <- 'C:/Research/actigraphy/data/amc_new/nonwear/'
dir.create(calibdir, showWarnings = FALSE)
dir.create(nonweardir, showWarnings = FALSE)

files <- list.files(path = indir, pattern='*.RData')
lapply(files, function(fname) {
    load(file.path(indir,fname))
    
    # Save calibration parameters
    calib <- data.frame(C$scale, C$offset, C$tempoffset)
    colnames(calib) <- c('scale','offset','tempoffset')
    calib_fname <- paste(substr(fname,6,nchar(fname)-10),'.csv',sep='')
    write.table(calib, file.path(calibdir,calib_fname), sep="\t", quote=FALSE, row.names=FALSE, col.names=TRUE)
    
    # Save nonwear info
    nonwear <- data.frame(M$metalong$timestamp, M$metalong$nonwearscore)
    colnames(nonwear) <- c('timestamp', 'nonwearscore')
    nonwear_fname <- paste(substr(fname,6,nchar(fname)-10),'.csv',sep='')
    write.table(nonwear, file.path(nonweardir,nonwear_fname), sep="\t", quote=FALSE, row.names=FALSE, col.names=TRUE)
})