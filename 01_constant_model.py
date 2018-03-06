import pandas as pd

CATEGORIES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
C = '0.0958,0.01,0.053,0.003,0.049,0.0088'

data = pd.read_csv('test.csv')

print('id,' + ','.join(CATEGORIES))
for ix, row in data.iterrows():
  print('%s,%s' % (row['id'], C))
