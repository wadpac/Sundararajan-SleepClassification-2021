library(GGIR)
library(rhdf5)

indir <- 'C:/Research/actigraphy/data/UPenn_Geneactiv/Axivity\ files/'
outdir <- 'C:/Research/actigraphy/data/UPenn_Geneactiv/raw_data/'
dir.create(outdir, showWarnings = FALSE)

max_num_pg <- 500000
files <- list.files(path = indir, pattern='*.cwa')
lapply(files, function(fname) {
  pg_start <- 0
  num_pg <- 20000
  tot_ndata <- 0
  print(paste('Started processing ',fname))
  out_fname <- paste(substr(fname,1,nchar(fname)-4),'.h5',sep='')
  out_fname <- file.path(outdir,out_fname)
  h5createFile(out_fname)
  h5createDataset(out_fname, 'timestamp', 300*max_num_pg, H5Sunlimited(), storage.mode='character', size=25)
  h5createDataset(out_fname, 'X', 300*max_num_pg, H5Sunlimited(), storage.mode='double')
  h5createDataset(out_fname, 'Y', 300*max_num_pg, H5Sunlimited(), storage.mode='double')
  h5createDataset(out_fname, 'Z', 300*max_num_pg, H5Sunlimited(), storage.mode='double')
  h5createDataset(out_fname, 'temp', 300*max_num_pg, H5Sunlimited(), storage.mode='double')
  h5createDataset(out_fname, 'battery', 300*max_num_pg, H5Sunlimited(), storage.mode='double')
  h5createDataset(out_fname, 'light', 300*max_num_pg, H5Sunlimited(), storage.mode='double')
  repeat{
    pg_end <- pg_start + num_pg
    D <- g.cwaread(file.path(indir,fname), pg_start, pg_end, desiredtz='America/New_York')
    # End if no data was read
    if ((nrow(D$data) == 0) || (is.null(D$data) == TRUE)) {
      break
    }
    ts <- as.character(as.POSIXct(D$data$time, origin='1970-01-01', tz='America/New_York'))
    ndata <- length(ts)
    st_idx <- tot_ndata + 1
    end_idx <- tot_ndata + ndata
    h5write(ts, out_fname, 'timestamp', index=list(st_idx:end_idx))
    h5write(D$data$x, out_fname, 'X', index=list(st_idx:end_idx))
    h5write(D$data$y, out_fname, 'Y', index=list(st_idx:end_idx))
    h5write(D$data$z, out_fname, 'Z', index=list(st_idx:end_idx))
    h5write(D$data$temp, out_fname, 'temp', index=list(st_idx:end_idx))
    h5write(D$data$battery, out_fname, 'battery', index=list(st_idx:end_idx))
    h5write(D$data$light, out_fname, 'light', index=list(st_idx:end_idx))
    print(pg_end)
    pg_start <- pg_end
    tot_ndata <- tot_ndata + ndata
  }
  h5set_extent(out_fname,'timestamp', tot_ndata)
  h5set_extent(out_fname,'X', tot_ndata)
  h5set_extent(out_fname,'Y', tot_ndata)
  h5set_extent(out_fname,'Z', tot_ndata)
  h5set_extent(out_fname,'temp', tot_ndata)
  h5set_extent(out_fname,'battery', tot_ndata)
  h5set_extent(out_fname,'light', tot_ndata)
  print(paste('Finished processing ',fname))
})