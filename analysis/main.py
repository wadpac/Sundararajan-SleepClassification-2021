import sys,os
from analysis import cv_classification_report, cv_hierarchical_classification_report

def main(argv):
  infile = argv[0] 
  mode = argv[1] 
  if mode != 'hierarchical':
    cv_classification_report(infile, mode)
  else:
    cv_hierarchical_classification_report(infile)

if __name__ == "__main__":
  main(sys.argv[1:])  
