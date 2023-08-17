import torch
from sentencepeice_tokenizer import SentencePieceTokenizer

tokenizer = SentencePieceTokenizer()

for i in range(10000):
  print(i)
  x = torch.tensor([i])
  print(tokenizer.decode(x.tolist()))