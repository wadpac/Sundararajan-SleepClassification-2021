import sys,os
import pandas as pd
from collections import Counter

def main(argv):
  featfile = argv[0]
  userfile = argv[1]
  dataset = argv[2]

  feat = pd.read_csv(featfile)
  users = pd.read_csv(userfile)
  if dataset == "Newcastle":
    users['id'] = ['MECSLEEP{:02d}'.format(int(user)) for user in users['id'].astype(str)]
    id_col = 'id'
    sex_col = 'Sex'; age_col = 'Age'; disorder_col = 'Disorder'
  elif dataset == "UPenn":
    id_col = 'ID'
    sex_col = 'Sex'; age_col = 'Age'
  elif dataset == 'AMC':
    id_col = 'study_id'
    sex_col = 'male'; age_col = 'age'; disorder_col = 'insomnia'
  users[id_col] = users[id_col].astype(str)

  print(len(set(feat['user'].astype(str))))
  dataset_users = list(set(feat['user'][feat['dataset'] == dataset].astype(str)))
  print('Num users = {:d}'.format(len(dataset_users)))

  print(sorted(set(dataset_users) - set(dataset_users).intersection(set(users[id_col]))))
  print(sorted(set(users[id_col]) - set(dataset_users).intersection(set(users[id_col]))))

  feat_users = list(set(feat['user'].astype(str)))
  sex = users[users[id_col].isin(feat_users)][sex_col]
  print(Counter(sex))
  age = users[users[id_col].isin(feat_users)][age_col]
  print('Age range: {:d} - {:d}'.format(int(age.min()), int(age.max())))
  if dataset != 'UPenn':
    disorder = users[users[id_col].isin(feat_users)][disorder_col]
    print(Counter(disorder))

if __name__ == "__main__":
  main(sys.argv[1:])
