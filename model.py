import torch
import math

class DecoderModel(torch.nn.Module):
  def __init__(self, max_seq_len, vocab_len, embedding_dimensions):
    super(DecoderModel, self).__init__()
    
    self.max_seq_len = max_seq_len
    self.vocab_len   = vocab_len
    self.embedding_dimensions = embedding_dimensions

    self.embedding    = torch.nn.Embedding(self.vocab_len, self.embedding_dimensions)
    self.pos_emb      = self.get_pos_matrix()

    self.add_and_norm = AddAndNorm(self.embedding_dimensions)

    self.attn_one = SelfAttention(self.max_seq_len, self.embedding_dimensions)
    self.attn_two = SelfAttention(self.max_seq_len, self.embedding_dimensions)
    self.feed_forward = FeedForward(self.embedding_dimensions)
    self.map_to_vocab = torch.nn.Linear(self.embedding_dimensions, self.vocab_len)

  def forward(self, x):
    emb = self.embedding(x)
    pos = self.pos_emb[0:x.shape[1], :]
    pos_emb_x = emb + pos

    attn_one = self.attn_one(pos_emb_x, x=x)
    attn_one_add_norm = self.add_and_norm(attn_one, pos_emb_x)

    attn_two = self.attn_two(attn_one_add_norm, x=x)
    attn_two_add_norm = self.add_and_norm(attn_two, attn_one_add_norm)
    
    res = self.feed_forward(attn_two_add_norm)
    out = self.map_to_vocab(res)

    return out

  def get_pos_matrix(self):
    store = torch.zeros(self.max_seq_len, self.embedding_dimensions)
    for pos in range(self.max_seq_len):
      for i in range(0, self.embedding_dimensions, 2):
        denominator = 10000 ** (2 * i / self.embedding_dimensions)
        store[pos, i] = math.sin(pos / denominator)
        if i + 1 < self.embedding_dimensions: store[pos, i + 1] = math.cos(pos / denominator)
    return store
  
class SelfAttention(torch.nn.Module):
    def __init__(self, max_seq_len, embedding_dimensions):
        super(SelfAttention, self).__init__()
        self.register_buffer('mask', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.key = torch.nn.Linear(embedding_dimensions, embedding_dimensions)
        self.qry = torch.nn.Linear(embedding_dimensions, embedding_dimensions)
        self.val = torch.nn.Linear(embedding_dimensions, embedding_dimensions)

    def forward(self, x_embeddings, x):
        key = self.key(x_embeddings)
        qry = self.qry(x_embeddings)
        val = self.val(x_embeddings)

        k_transpose = key.permute(0,2,1)
        att = torch.bmm(qry, k_transpose)
        msk = self.mask[0:x.shape[1], 0:x.shape[1]]
        batch_msk = msk.unsqueeze(0).expand(att.size())
        att = att.masked_fill(batch_msk == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=1)
        res = torch.bmm(att, val)
        return res
   
class AddAndNorm(torch.nn.Module):
    def __init__(self, embedding_dimensions):
        super(AddAndNorm, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(embedding_dimensions)

    def forward(self, sublayer_output, residual_x):
        return self.layer_norm(sublayer_output + residual_x)
    
class FeedForward(torch.nn.Module):
   def __init__(self, embedding_dimensions):
      super(FeedForward, self).__init__()
      self.feed_forward = torch.nn.Linear(embedding_dimensions, embedding_dimensions)
      self.relu = torch.nn.ReLU()

   def forward(self, x):
      x = self.feed_forward(x)
      x = self.relu(x)
      x = self.feed_forward(x)
      return x
    

   