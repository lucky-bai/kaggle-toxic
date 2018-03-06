import pandas as pd

CATEGORIES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

data = pd.read_csv('test.csv')

print('id,' + ','.join(CATEGORIES))
for ix, row in data.iterrows():
  if 'fuck' in row['comment_text']:
    print('%s,1,1,1,1,1,1' % row['id'])
  else:
    print('%s,0,0,0,0,0,0' % row['id'])
