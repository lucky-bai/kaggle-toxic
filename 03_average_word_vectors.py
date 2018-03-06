from collections import OrderedDict
import spacy
import numpy as np
import pandas as pd
import sys
from linguistic_features import compute_linguistic_features

DEBUG = False
PROCESS_MODE = sys.argv[1]

print('Loading Spacy')
if not DEBUG:
  nlp = spacy.load('en_core_web_md', disable = ['tagger', 'parser', 'ner'])

print('Loading data')
data_train = pd.read_csv(PROCESS_MODE + '.csv')

def text_to_vector(text):
  if DEBUG:
    d = OrderedDict()
    d['vec0'] = 0.5
    d['vec1'] = 0.6
    return d

  tokens = nlp(text)
  d = OrderedDict()
  for ix, vs in enumerate(tokens.vector.tolist()):
    feat_name = 'vec%d' % ix
    d[feat_name] = vs

  d.update(compute_linguistic_features(text))
  
  return d


def main():
  #example_text = 'I eat cats for breakfast!'
  #text_to_vector(example_text)

  out = []
  for ix, row in data_train.iterrows():
    if ix % 100 == 0:
      print('Row:', ix)

    d = OrderedDict()
    d['id'] = row['id']
    if PROCESS_MODE == 'train':
      d['toxic'] = row['toxic']
      d['severe_toxic'] = row['severe_toxic']
      d['obscene'] = row['obscene']
      d['threat'] = row['threat']
      d['insult'] = row['insult']
      d['identity_hate'] = row['identity_hate']
    text = row['comment_text']
    d.update(text_to_vector(text))
    out.append(d)

  pd.DataFrame(out).to_csv(PROCESS_MODE + '_features.csv', index = False)
  


main()


