import torch
from torch.nn.utils.rnn import pad_sequence

torch.manual_seed(42)

def pad(sequences, batch_first=True, padding_value=0.0):
    return pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)

class TinyStoriesDataset(torch.utils.data.Dataset):
  def __init__(self, tokenizer):
    self.data_filename = "TinyStories-1000.txt"
    f = open("data/" + self.data_filename, 'r')
    self.story_lines = f.read().split('\n')
    self.tokenizer = tokenizer
    print("Number of stories: ", len(self.story_lines))
    f.close()

  def __len__(self):
    return len(self.story_lines)

  def __getitem__(self, idx):
    story_line = self.story_lines[idx]
    return torch.tensor(self.tokenizer.encode(story_line))
  
  def get_data_filename(self):
    return self.data_filename