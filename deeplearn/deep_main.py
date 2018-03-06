import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.autograd import Variable
import spacy
import pdb


# Size of hidden layer of LSTM
HIDDEN_SIZE = 1000

# Size of word embeddings
WORDVEC_SIZE = 300

# Number of label classes
NUM_LABELS = 6

BATCH_SIZE = 200



class DeepClassifier(nn.Module):
  """Predict 6 classes using the same shared hidden layer"""

  def __init__(self):
    super(DeepClassifier, self).__init__()

    self.nlp = spacy.load('en_core_web_md')
    
    self.lstm = nn.LSTM(
      input_size = WORDVEC_SIZE,
      hidden_size = HIDDEN_SIZE,
      batch_first = True,
    )
    self.multi_classifier = nn.Linear(HIDDEN_SIZE, NUM_LABELS)

  def _sigmoid(self, x):
    return 1 / (1 + np.exp(-x))


  def forward(self, text):

    # Feed embeddings through LSTM
    wordvec_sequence = []
    tokens = self.nlp(text)
    for tok in tokens:
      wordvec_sequence.append(tok.vector)
    wordvec_sequence = np.vstack(wordvec_sequence)
    wordvec_sequence = Variable(torch.Tensor(wordvec_sequence)).cuda()
    wordvec_sequence = wordvec_sequence.unsqueeze(0)

    # Get last output layer
    lstm_out, _ = self.lstm(wordvec_sequence)
    lstm_out = lstm_out[:, -1]

    # Feed LSTM output to 6 binary classifiers
    probs = self.multi_classifier(lstm_out)

    # Sigmoid to transform to [0, 1]
    probs = torch.sigmoid(probs)

    return probs



def main():
  #text = 'I eat cats for breakfast!'
  #truth = Variable(torch.Tensor([1, 0, 0, 0, 0, 1])).cuda()

  train_data = pd.read_csv('../train.csv')

  model = DeepClassifier().cuda()
  optimizer = optim.Adam(model.parameters())
  loss_fn = nn.MSELoss()

  # Just do one epoch through the data for now
  batch_loss = 0
  for ix, row in train_data.iterrows():
    text = row['comment_text']
    truth = Variable(torch.Tensor(row[2:8])).cuda()

    if ix % BATCH_SIZE == 0:
      optimizer.zero_grad()

    out = model(text)
    loss = loss_fn(out, truth)
    batch_loss += loss

    if ix % BATCH_SIZE == 0:
      batch_loss.backward()
      optimizer.step()
      print(ix, float(batch_loss))
      batch_loss = 0

  torch.save(model, 'model.t7')


main()
