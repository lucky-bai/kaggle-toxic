import pandas as pd

data_test = pd.read_csv('../test.csv')

CATEGORIES = 'toxic severe_toxic obscene threat insult identity_hate'.split()

df = pd.DataFrame()

df['id'] = data_test['id']
for ix, c in enumerate(CATEGORIES):
  with open('o' + str(ix+1) + '.txt') as f:
    col = []
    for line in f.readlines():
      a, b = line.split()
      b = float(b)
      if a == '__label__yes':
        col.append(b)
      else:
        col.append(1-b)

  df[c] = col

df.to_csv('submission.csv', index = False)
