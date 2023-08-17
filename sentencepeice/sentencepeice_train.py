import sentencepiece as spm

# There is also a more pytorch way of doing this. 

# train sentencepiece model from `botchan.txt` and makes `m.model` and `m.vocab`
# `m.vocab` is just a reference. not used in the segmentation.
spm.SentencePieceTrainer.train('--input=../data/TinyStories-1000.txt --model_prefix=m --vocab_size=4379')

# makes segmenter instance and loads the model file (m.model)
sp = spm.SentencePieceProcessor()
sp.load('m.model')

# encode: text => id
print(sp.encode_as_pieces('This is a test'))
print(sp.encode_as_ids('This is a test'))

# decode: id => text
print(sp.decode_pieces(['▁This', '▁is', '▁a', '▁t', 'est']))
print(sp.decode_ids([209, 31, 9, 375, 586]))

# returns vocab size
print(sp.get_piece_size())

# id <=> piece conversion
print(sp.id_to_piece(209))
print(sp.piece_to_id('▁This'))

# returns 0 for unknown tokens (we can change the id for UNK)
print(sp.piece_to_id('__MUST_BE_UNKNOWN__'))

# <unk>, <s>, </s> are defined by default. Their ids are (0, 1, 2)
# <s> and </s> are defined as 'control' symbol.
for id in range(3):
  print(sp.id_to_piece(id), sp.is_control(id))