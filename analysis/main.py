import sys,os
from analysis import cv_classification_report

def main(argv):
  infile = argv[0] 
  mode = argv[1] 
  cv_classification_report(infile, mode)

if __name__ == "__main__":
  main(sys.argv[1:])  
