from datetime import datetime
from typing import Callable, Dict, List, Tuple
from tinygrad import Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_state_dict, safe_save, get_parameters
from tinygrad.nn import Linear, BatchNorm
from tinygrad.dtype import dtypes

import numpy as np
import re


def vectorize(text: str) -> Tuple[np.ndarray, List[str], Dict[str, int], List[int]]:
  words = re.split(r"[\W+]", text.lower())
  tok2word = list(set(words))  # finds the set of distinct words
  tok2word = [word for word in tok2word if word]  # remove empty word from list
  word2tok = {item: index for index, item in enumerate(tok2word)}  # maps numbers to words
  tokens: List[int] = [word2tok[word] for word in words if word]  # list of words -> list of numbers
  vectors = np.vstack([one_hot(token, len(tok2word)) for token in tokens])  # one-hot encoded tokens
  return (vectors, tok2word, word2tok, tokens)


def one_hot(num, max):
  vec = np.zeros(max)
  vec[num] = 1
  return vec


# raw_data = "If the specified pattern is not found inside the target string, then the string is not split in any way,
# but the split method still generates a list since this is the way it’s designed. However, the list contains just one
# element, the target string itself."
with open("data.txt", "r") as file:
  raw_data = file.read()

(vectors, tok2word, word2tok, tokens) = vectorize(raw_data)
vocab_size = len(tok2word)

n = 3  # ngram size
input_size = n * vocab_size

X = Tensor(np.vstack([np.hstack(vectors[i:i + n]) for i in range(len(vectors) - n + 1)]),
           dtype=dtypes.float)  # bag of words
Y = Tensor(tokens[n:], dtype=dtypes.float)  # categ. cross entr. expects classes, so we use the token indices.


# Y = Tensor(vectors[n:]) # better results may be achieved by using one hot encoding on the labels with another loss.

class Dropout:
  def __init__(self, p: float):
    self.p = p

  def __call__(self, x) -> Tensor: return x.dropout(self.p)


@TinyJit
class TinyNet:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      Linear(input_size, 128, bias=True), Tensor.relu,
      BatchNorm(128),
      Linear(128, 128, bias=True), Tensor.relu,
      Dropout(.2),
      Linear(128, 128, bias=True), Tensor.relu,
      Linear(128, vocab_size, bias=True)
    ]

  def __call__(self, x: Tensor) -> Tensor: return x.sequential(self.layers)


net = TinyNet()

# optimizer
opt = Adam(get_parameters(net), lr=3e-3)
batch_size = 64

# training loop
with Tensor.train():
  for step in range(10000):
    samples = Tensor.randint(batch_size, high=X.shape[0])

    labels = Y[samples]
    out = net(X[samples])

    loss = out.sparse_categorical_crossentropy(labels)

    opt.zero_grad()
    loss.backward()
    opt.step()

    # calculate accuracy
    pred = out.argmax(axis=-1)
    acc = (pred == labels).mean()

    if step % 100 == 0:
      print(f"Step {step + 1} | Loss: {loss.item()} | Accuracy: {acc.numpy()}")

# persist model

name = datetime.now().strftime("trained/chat_%Y_%m_%d__%H_%M_%S.safetensors")
state_dict = get_state_dict(net)
safe_save(state_dict, name)


# here's the inference part

def decode(vec, tok2word):
  token = np.argmax(vec.numpy())
  return tok2word[token]  # finds largest element in vector and decodes using tok2word


print("vocab size: {}".format(vocab_size))

while True:
  try:
    prompt = input("\n\ntype something: \n")
    words = re.split(r"[\W+]", prompt.lower())

    for _ in range(0, 32):
      tokens = [word2tok[word] for word in words[-n:]]
      encoded = Tensor(np.hstack([one_hot(token, vocab_size) for token in tokens]), dtype=dtypes.float)
      
      # usually, you can just pipe batches or single samples through tinynet without
      # having to modify the code, however, the batchnorm layer expects data to be passed
      # in batches, so for inference mode, we just tack on a dimension to the left,
      # so it gets data of shape (1, 128) instead of (128), which would trip it up
      encoded = encoded.unsqueeze(0)
      out = net(encoded)
      res = decode(out, tok2word)
      words.append(res)
      print(res, end=" ")

  except KeyboardInterrupt:
    break

  except KeyError as e:
    print("\n  An error occurred:")
    print("    Unknown token: {}".format(e))
