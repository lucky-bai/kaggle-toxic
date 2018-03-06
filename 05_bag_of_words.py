from collections import OrderedDict
from collections import Counter
import numpy as np
import pandas as pd
import string

TEST_COMMENT = """
Yes yes and yes! I don't know the songs directly by name, but if it were to play, I know I'd dance my ass off cause songs like those are classics!!! D I was at a dragqueen bar this weekend, and ""Countdown"" came on, and me and my friend got up and danced all over the place! Hahahahahahaaa!!! I might be a white-boy, but i sure as hell don't dance like one! ;D  STACK STACK STACK "
"""

NUM_FEATURES = 500

def segment_to_words(text):
  text = text.translate(str.maketrans('', '', string.punctuation))
  text = text.lower().split()
  return text


print('Reading data')
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

print('Counting words')
all_words = []
for ix, row in data_train.iterrows():
  comment_text = row['comment_text']
  all_words.extend(segment_to_words(comment_text))

# Construct feature dict
ix_word_dict = {}
ctr = Counter(all_words)
for ix, (word, _) in enumerate(ctr.most_common(NUM_FEATURES)):
  ix_word_dict[ix] = word


def make_features(is_train):
  data_train_test = data_train if is_train else data_test
  out = []
  for ix, row in data_train_test.iterrows():
    if ix % 100 == 0:
      print('Row', ix)
    d = OrderedDict()
    d['id'] = row['id']
    if is_train:
      d['toxic'] = row['toxic']
      d['severe_toxic'] = row['severe_toxic']
      d['obscene'] = row['obscene']
      d['threat'] = row['threat']
      d['insult'] = row['insult']
      d['identity_hate'] = row['identity_hate']
    comment_text = row['comment_text']
    words = segment_to_words(comment_text)
    d['tokens'] = len(words)
    for ix in range(NUM_FEATURES):
      d['w%d' % ix] = words.count(ix_word_dict[ix]) / max(len(words), 1)
    out.append(d)
  if is_train:
    pd.DataFrame(out).to_csv('train_features.csv', index = False)
  else:
    pd.DataFrame(out).to_csv('test_features.csv', index = False)

    
#make_features(True)
make_features(False)