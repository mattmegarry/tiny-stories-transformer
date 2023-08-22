import torch
import math

dropout_rate = 0.1
encoder_layers = 6
decoder_layers = 6

class Transformer(torch.nn.Module):
  def __init__(self, max_seq_len, vocab_len, embedding_dimensions):
    super(Transformer, self).__init__()
    self.max_seq_len = max_seq_len
    self.vocab_len   = vocab_len
    self.embedding_dimensions = embedding_dimensions

    self.embedding    = torch.nn.Embedding(self.vocab_len, self.embedding_dimensions)
    self.pos_emb      = self.get_pos_matrix()
    self.dropout      = torch.nn.Dropout(dropout_rate)
    self.encoder_blocks = torch.nn.ModuleList([EncoderBlock(self.max_seq_len, self.embedding_dimensions, num_heads=4) for _ in range(encoder_layers)])
    self.decoder_blocks = torch.nn.ModuleList([DecoderBlock(self.max_seq_len, self.embedding_dimensions, num_heads=4) for _ in range(decoder_layers)])
    self.final_layer_norm = torch.nn.LayerNorm(self.embedding_dimensions)
    self.map_to_vocab = torch.nn.Linear(self.embedding_dimensions, self.vocab_len)

  def forward(self, source, target=None):
    emb = self.embedding(source)
    pos = self.pos_emb[0:source.shape[1], :]
    source = emb + pos
    source = self.dropout(source)

    out = None
    if target is not None:
      for encoder_block in self.encoder_blocks: source = encoder_block(source)

      emb = self.embedding(target.long())
      pos = self.pos_emb[0:target.shape[1], :]
      target = emb + pos
      target = self.dropout(target)

      for decoder_block in self.decoder_blocks: out = decoder_block(source, target)
    else:
      for decoder_block in self.decoder_blocks: out = decoder_block(source) 
    
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
  
  def translate(self, src, num=20):
    self.eval()
    tgt = src # torch.tensor([[1]])
    print(tgt)
    for _ in range(num):
      with torch.no_grad():
        out = self(src, tgt)
        out = out[:, -1, :]
        nxt = torch.argmax(out, dim=-1, keepdim=True)
        print(nxt)
        if nxt.item() == 2: break
        tgt = torch.cat([tgt, nxt], dim=1)
    self.train()
    return tgt

class EncoderBlock(torch.nn.Module):
    def __init__(self, max_seq_len, embedding_dimensions, num_heads):
      super(EncoderBlock, self).__init__()
      self.sublayer = NonCausalSublayer(max_seq_len, embedding_dimensions, num_heads)

    def forward(self, x):
      output = self.sublayer(x)
      return output

class DecoderBlock(torch.nn.Module):
    def __init__(self, max_seq_len, embedding_dimensions, num_heads):
      super(DecoderBlock, self).__init__()
      self.embedding_dimensions = embedding_dimensions

      self.encoder_qkv_projection = torch.nn.Linear(embedding_dimensions, 3 * embedding_dimensions)
      self.decoder_merge = torch.nn.Linear(3 * embedding_dimensions, embedding_dimensions)

      self.masked_sublayer = CausalSublayer(max_seq_len, embedding_dimensions, num_heads)
      self.unmasked_sublayer = NonCausalSublayer(max_seq_len, embedding_dimensions, num_heads)

    def forward(self, source, target=None):
      if target is not None:
        causal_output = self.masked_sublayer(target) # This might be wrong
      else:
        causal_output = self.masked_sublayer(source) 

      if target is not None:
        source = self.encoder_qkv_projection(source)
        source_q, source_k, source_v = source.split(self.embedding_dimensions, dim=-1)
        print(causal_output.shape)
        print(source_k.shape)
        print(source_v.shape)
        non_causal_input = torch.cat([causal_output, source_k, source_v], dim=-1)
        non_causal_input = self.decoder_merge(non_causal_input)
        output = self.unmasked_sublayer(non_causal_input)
      else:
        output = self.unmasked_sublayer(causal_output)
      return output
    
class NonCausalSublayer(torch.nn.Module):
    def __init__(self, max_seq_len, embedding_dimensions, num_heads):
      super(NonCausalSublayer, self).__init__()
      self.layer_norm = torch.nn.LayerNorm(embedding_dimensions)
      self.attn = MultiHeadAttention(max_seq_len, embedding_dimensions, num_heads, is_causal=False)
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

class CausalSublayer(torch.nn.Module):
    def __init__(self, max_seq_len, embedding_dimensions, num_heads):
      super(CausalSublayer, self).__init__()
      self.layer_norm = torch.nn.LayerNorm(embedding_dimensions)
      self.attn = MultiHeadAttention(max_seq_len, embedding_dimensions, num_heads, is_causal=True)
      self.attn_add_and_norm = AddAndNorm(embedding_dimensions)

    def forward(self, x):
      residual_x = x
      output = self.layer_norm(x)
      output = self.attn(output)
      output = self.attn_add_and_norm(output, residual_x)
      return output
    
class MultiHeadAttention(torch.nn.Module):
   def __init__(self, max_seq_len, embedding_dimensions, num_heads, is_causal=False):
      super(MultiHeadAttention, self).__init__()
      self.is_causal = is_causal
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
      att = att if self.is_causal == False else att.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
      att = torch.nn.functional.softmax(att, dim=-1)
      y = att @ values
      y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dimensions)

      output = self.output_projection(y)
      return output

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
    

   