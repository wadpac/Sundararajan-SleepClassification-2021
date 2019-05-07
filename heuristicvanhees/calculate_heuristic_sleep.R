# rm(list=ls())
# Use epoch level data to calculate sleep classification  with heuristic method (van Hees 2015 PLOSONE)
library(data.table)
fname5 = "/media/sf_sharedfolder/LRAP_epochdata/sleep_data_5.0s.csv"
fname30 = "/media/sf_sharedfolder/LRAP_epochdata/sleep_data_30.0s.csv"
outputfile5 =  "/media/sf_sharedfolder/LRAP_epochdata/sleep_data_5.0s_heuristic.csv"
outputfile30 =  "/media/sf_sharedfolder/LRAP_epochdata/sleep_data_30.0s_heuristic.csv"


# parameters for sleep detection algorithm
timethreshold = 5 # in minutes
anglethreshold = 5 # in degrees

conv2factor = function(x) {
  # convert column heuristic to factor
  x$heuristic = as.factor(x$heuristic)
  LV = levels(x$heuristic)
  if (LV[1] == "0" & LV[2] == "1") {
    levels(x$heuristic) = c("Wake", "Sleep")
  } else if (LV[2] == "0" & LV[1] == "1") {
    levels(x$heuristic) = c("Sleep", "Wake")
  }
  return(x)
}

for (epochsize in c(5,30)) { # epoch length in seconds
  # load data
  if (epochsize == 5) {
    D = data.table::fread(fname5)
  } else {
    D = data.table::fread(fname30)
  }
  # remove rows with unwanted labels
  unwantedlabels = c("Supine","Left","Right", "Prone", "Lights On", "Lights Off", "Bathroom In", "Bathroom Out")
  D = D[-which(D$label %in% unwantedlabels == TRUE),] # remove all rows that are not epochs
  # add dummy column for heuristic output
  D$heuristic = 2
  # identify unique dataset-user combinations
  datasets = unique(D[,c("dataset","user")])
  for (i in 1:nrow(datasets)) { # loop over unique dataset-user combinations
    dataset = datasets$dataset[i]
    user = datasets$user[i]
    cat(paste0("\n",dataset,"_",user))
    # get data segment
    SleepWakeScore = c() # initialise output variable wiht the SleepWakeScores
    segment = which(D$dataset == dataset & D$user == user)
    Dseg = D[segment,]
    # extract angle
    angle = Dseg$angz_mean
    angle[which(is.na(angle) == T)] = 0 # possibly redundant step
    SleepWakeScore = rep(0,length(angle))
    AngleChangePoint = which(abs(diff(angle)) > anglethreshold)
    sib_indices = c() # sustained inactivity bouts indices
    if (length(AngleChangePoint) > 1) {
      sib_indices = which(diff(AngleChangePoint) > (timethreshold*(60/epochsize))) #less than once per timethreshold minutes
    }
    if (length(sib_indices) > 0) {
      for (gi in 1:length(sib_indices)) {
        SleepWakeScore[AngleChangePoint[sib_indices[gi]]:AngleChangePoint[sib_indices[gi]+1]] = 1 #periods with no angle change
      }
    }
    # store back in original data.frame
    if (length( SleepWakeScore ) > 0) {
      D[segment,"heuristic"] = SleepWakeScore
    }
  }
  D = conv2factor(D)
  # Save output to files
  if (epochsize == 5) {
    write.csv(D,file = outputfile5,row.names = FALSE)
  } else {
    write.csv(D,file = outputfile30,row.names = FALSE)
  }
}