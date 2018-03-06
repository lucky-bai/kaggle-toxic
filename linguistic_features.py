from collections import OrderedDict
import string
import re
from badwords import badwords

count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))

def div(a, b):
  if b == 0:
    return 0
  else:
    return a/b


def compute_linguistic_features(text):
  words = text.split()
  wordslower = text.lower().split()

  od = OrderedDict()

  # Length metrics
  od['characters'] = len(text)
  od['tokens'] = len(words)

  # Count of exclamation marks
  od['exclamation_count'] = text.count('!')
  od['exclamation_ratio'] = div(text.count('!'), len(text))

  # Profanity count
  od['contains_profanity'] = (count(wordslower, badwords) > 0)
  od['profanity_count'] = count(wordslower, badwords)
  od['profanity_ratio'] = div(count(wordslower, badwords), len(words))

  # Count of punctuation, whitespace
  od['punctuation_count'] = count(text, string.punctuation)
  od['punctuation_ratio'] = div(count(text, string.punctuation), len(text))
  od['whitespace_count'] = count(text, string.whitespace)
  od['whitespace_ratio'] = div(count(text, string.whitespace), len(text))

  # Count of upper/lower letters
  od['upper_count'] = count(text, string.ascii_uppercase)
  od['upper_ratio'] = div(count(text, string.ascii_uppercase), len(text))
  od['lower_count'] = count(text, string.ascii_lowercase)
  od['lower_ratio'] = div(count(text, string.ascii_lowercase), len(text))

  # Completely uppercase
  od['allcaps_count'] = len(list(filter(lambda w: len(w) > 1 and w.isupper(), words)))
  od['allcaps_ratio'] = div(len(list(filter(lambda w: len(w) > 1 and w.isupper(), words))), len(words))

  # Contains URL
  od['contains_url'] = ('http://' in text) or ('https://' in text)

  # Contains IP address
  od['contains_ip'] = bool(re.search(r'\d+\.\d+\.\d+\.\d+', text))

  return od


#STR = 'I like to eat cats!!!!... REALLY 6.32.6.5 FUCKING'
#print(compute_linguistic_features(STR))

