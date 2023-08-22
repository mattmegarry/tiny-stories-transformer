import torch
import math

dropout_rate = 0.1
decoder_layers = 6

class DecoderModel(torch.nn.Module):
  def __init__(self, max_seq_len, vocab_len, embedding_dimensions):
    super(DecoderModel, self).__init__()
    
    self.max_seq_len = max_seq_len
    self.vocab_len   = vocab_len
    self.embedding_dimensions = embedding_dimensions

    self.embedding    = torch.nn.Embedding(self.vocab_len, self.embedding_dimensions)
    self.pos_emb      = self.get_pos_matrix()
    self.dropout      = torch.nn.Dropout(dropout_rate)
    self.blocks = torch.nn.ModuleList([Block(self.max_seq_len, self.embedding_dimensions, 4) for _ in range(decoder_layers)])
    self.final_layer_norm = torch.nn.LayerNorm(self.embedding_dimensions)
    self.map_to_vocab = torch.nn.Linear(self.embedding_dimensions, self.vocab_len)

  def forward(self, x):
    emb = self.embedding(x)
    pos = self.pos_emb[0:x.shape[1], :]
    pos_emb_x = emb + pos
    out = self.dropout(pos_emb_x)

    for block in self.blocks: out = block(out)
    out = self.final_layer_norm(out)
    out = self.map_to_vocab(out)

    return out

  def get_pos_matrix(self):
    store = torch.zeros(self.max_seq_len, self.embedding_dimensions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = store.to(device)
    for pos in range(self.max_seq_len):
      for i in range(0, self.embedding_dimensions, 2):
        denominator = 10000 ** (2 * i / self.embedding_dimensions)
        store[pos, i] = math.sin(pos / denominator)
        if i + 1 < self.embedding_dimensions: store[pos, i + 1] = math.cos(pos / denominator)
    return store



class MultiHeadAttention(torch.nn.Module):
   def __init__(self, max_seq_len, embedding_dimensions, num_heads):
      super(MultiHeadAttention, self).__init__()
      self.max_seq_len = max_seq_len
      assert embedding_dimensions % num_heads == 0, "Embedding dimensions must be divisible by number of heads"
      
      self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len))

      self.num_heads = num_heads
      self.embedding_dimensions = embedding_dimensions
      self.head_size = embedding_dimensions // num_heads

      self.qkv_projection = torch.nn.Linear(embedding_dimensions, 3 * embedding_dimensions)
      self.output_projection = torch.nn.Linear(embedding_dimensions, embedding_dimensions)

   def forward(self, x_embeddings):
      batch_size, seq_len, _ = x_embeddings.size()

      q, k, v  = self.qkv_projection(x_embeddings).split(self.embedding_dimensions, dim=2)
      queries = q.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
      keys = k.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
      values = v.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

      att = (queries @ keys.transpose(-2, -1)) * (1.0 / math.sqrt(keys.size(-1)))
      att = att.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
      att = torch.nn.functional.softmax(att, dim=-1)
      y = att @ values
      y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dimensions)

      output = self.output_projection(y)
      return output

class Block(torch.nn.Module):
    def __init__(self, max_seq_len, embedding_dimensions, num_heads):
      super(Block, self).__init__()
      self.layer_norm = torch.nn.LayerNorm(embedding_dimensions)
      self.attn = MultiHeadAttention(max_seq_len, embedding_dimensions, num_heads)
      self.attn_add_and_norm = AddAndNorm(embedding_dimensions)
      self.feed_forward = FeedForward(embedding_dimensions)
      self.feed_forward_add_and_norm = AddAndNorm(embedding_dimensions)

    def forward(self, x):
      residual_x = x
      output = self.layer_norm(x)
      output = self.attn(output)
      residual_ff_x = self.attn_add_and_norm(output, residual_x)
      output = self.feed_forward(residual_ff_x)
      output = self.feed_forward_add_and_norm(output, residual_ff_x)
      return output

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
   def __init__(self, embedding_dimensions, ff_expansion=4):
      super(FeedForward, self).__init__()
      self.context_fully_connected = torch.nn.Linear(embedding_dimensions, embedding_dimensions * ff_expansion)
      self.relu = torch.nn.ReLU()
      self.context_projection = torch.nn.Linear(embedding_dimensions * ff_expansion, embedding_dimensions)
      self.dropout = torch.nn.Dropout(dropout_rate)

   def forward(self, x):
      x = self.context_fully_connected(x)
      x = self.relu(x)
      x = self.context_projection(x)
      x = self.dropout(x)
      return x
    

   