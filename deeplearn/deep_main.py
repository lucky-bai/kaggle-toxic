import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.autograd import Variable
from collections import OrderedDict
import spacy
import statistics
import pdb


# Size of hidden layer of LSTM
HIDDEN_SIZE = 1000

# Size of word embeddings
WORDVEC_SIZE = 300 + 1
END_MARKER = np.zeros(WORDVEC_SIZE)
END_MARKER[-1] = 1

# Number of label classes
NUM_LABELS = 6

BATCH_SIZE = 500
NUM_EPOCHS = 4

CATEGORIES = 'toxic severe_toxic obscene threat insult identity_hate'.split()



class DeepClassifier(nn.Module):
  """Predict 6 classes using the same shared hidden layer"""

  def __init__(self):
    super(DeepClassifier, self).__init__()

    self.nlp = spacy.load('en_core_web_md', disable = ['ner', 'parser', 'tagger'])
    
    self.lstm = nn.LSTM(
      input_size = WORDVEC_SIZE,
      hidden_size = HIDDEN_SIZE,
      batch_first = True,
    )

    # Put two layers before classification
    self.multi_classifier = nn.Sequential(
      nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
      nn.Dropout(p = 0.5),
      nn.ReLU(),
      nn.Linear(HIDDEN_SIZE, NUM_LABELS),
    )


  def forward(self, text):

    # Feed embeddings through LSTM
    wordvec_sequence = []
    tokens = self.nlp(text)
    for tok in tokens:
      v = tok.vector
      v = np.append(v, [0])
      wordvec_sequence.append(v)
    wordvec_sequence.append(END_MARKER)

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



def do_train():
  train_data = pd.read_csv('../train.csv')

  model = DeepClassifier().cuda()
  model.train()
  optimizer = optim.Adam(model.parameters(), lr = 0.0001)
  loss_fn = nn.MSELoss()
  torch.save(model.state_dict(), 'model.t7')

  for epoch in range(NUM_EPOCHS):
    batch_loss = 0
    for ix, row in train_data.iterrows():
      text = row['comment_text']
      truth = Variable(torch.Tensor(row[2:8])).cuda()

      if ix > 0 and ix % BATCH_SIZE == 0:
        optimizer.zero_grad()

      out = model(text)
      loss = loss_fn(out, truth)
      batch_loss += loss

      if ix > 0 and ix % BATCH_SIZE == 0:
        batch_loss.backward()
        optimizer.step()
        print('Epoch %d, iter %d, loss %0.9f' % (epoch, ix, float(batch_loss)))
        batch_loss = 0

    torch.save(model.state_dict(), 'model_epoch_%d.t7' % epoch)


def do_train_validation():
  """Compute training loss for model"""
  train_data = pd.read_csv('../train.csv')

  model = DeepClassifier().cuda()
  model.eval()
  model.load_state_dict(torch.load('model_lstm_lb_9433.t7'))
  loss_fn = nn.MSELoss()

  losses = []
  for ix, row in train_data.iterrows():
    if ix % 100 == 1:
      # Feel free to stop this whenever
      print(ix, statistics.mean(losses))

    text = row['comment_text']
    truth = Variable(torch.Tensor(row[2:8])).cuda()

    out = model(text)
    loss = float(loss_fn(out, truth))
    losses.append(loss)



def do_test():
  test_data = pd.read_csv('../test.csv')

  model = DeepClassifier().cuda()
  model.eval()
  model.load_state_dict(torch.load('model_epoch_3.t7'))

  out = []
  for ix, row in test_data.iterrows():
    if ix % 100 == 0:
      print(ix)

    text = row['comment_text']
    od = OrderedDict()
    od['id'] = row['id']
    model_out = model(text)
    for ix, ct in enumerate(CATEGORIES):
      od[ct] = float(model_out[0, ix])
    out.append(od)

  pd.DataFrame(out).to_csv('submission.csv', index = False)



do_train()
#do_train_validation()
#do_test()
