library(GGIR)

indir <- 'C:/Research/actigraphy/data/amc_new/amc_extracted/wrap/'
outdir <- 'C:/Research/actigraphy/data/amc_new/'
fileType = '*.bin'
dir.create(outdir, showWarnings = FALSE)

files <- list.files(path = indir, pattern=fileType)
nfiles <- length(files)
g.part1(datadir=indir, outputdir=outdir, f0=1, f1=nfiles)