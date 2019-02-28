library(GGIR)

indir <- 'C:/Research/actigraphy/data/UPenn_Geneactiv/Axivity\ files/'
outdir <- 'C:/Research/actigraphy/data/UPenn_Geneactiv/'
fileType = '*.cwa'
dir.create(outdir, showWarnings = FALSE)

files <- list.files(path = indir, pattern=fileType)
nfiles <- length(files)
g.part1(datadir=indir, outputdir=outdir, f0=1, f1=nfiles)