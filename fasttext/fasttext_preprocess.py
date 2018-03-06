# Prepare input for fasttext
import pandas as pd
import sys

mode = sys.argv[1]

if mode == 'test':
  data_test = pd.read_csv('../test.csv')
  for ix, row in data_test.iterrows():
    comment = row['comment_text']
    comment = comment.replace('\n', ' ')
    print(comment)

else:
  data_train = pd.read_csv('../train.csv')

  for ix, row in data_train.iterrows():
    labels = []
    if row[mode]:
      labels.append('__label__yes')
    else:
      labels.append('__label__no')
    
    comment = row['comment_text']
    comment = comment.replace('\n', ' ')
    print(' '.join(labels), comment)
