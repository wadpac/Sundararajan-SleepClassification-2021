import sys,os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def main(argv):
  infile = argv[0]

  df = pd.read_csv(infile)
  #df = df[df['dataset'] == 'UPenn'].reset_index()
  df['binary_label'] = df['label']
  df.loc[df['label'] == 'NREM 1','binary_label'] = 'Sleep'
  df.loc[df['label'] == 'NREM 2','binary_label'] = 'Sleep'
  df.loc[df['label'] == 'NREM 3','binary_label'] = 'Sleep'
  df.loc[df['label'] == 'REM','binary_label'] = 'Sleep'
  
  print('#samples: %d' % len(df))
  print('#Wake: %d' % len(df[df['label'] == 'Wake']))
  print('#NREM 1: %d' % len(df[df['label'] == 'NREM 1']))
  print('#NREM 2: %d' % len(df[df['label'] == 'NREM 2']))
  print('#NREM 3: %d' % len(df[df['label'] == 'NREM 3']))
  print('#REM: %d' % len(df[df['label'] == 'REM']))

  y_true = df['binary_label']
  y_pred = df['heuristic']
  precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, 
                                     labels=['Wake','Sleep'], average='macro')
  print('Precision = %0.4f' % (precision*100.0))
  print('Recall = %0.4f' % (recall*100.0))
  print('F-score = %0.4f' % (fscore*100.0))
  print(classification_report(y_true,y_pred,digits=4))

if __name__ == "__main__":
  main(sys.argv[1:])
