from torchtext.data.functional import load_sp_model

class SentencePieceTokenizer:
    def __init__(self):
        self.sp_model = load_sp_model("sentencepeice/m.model")
    
    def encode(self, text):
        return self.sp_model.EncodeAsIds(text)
    
    def encode_as_pieces(self, text):
        return self.sp_model.EncodeAsPieces(text)

    def decode(self, tokens):
        return self.sp_model.DecodeIds(tokens)
    
    def get_vocab_size(self):
        return self.sp_model.GetPieceSize()

# Just for testing...
"""
tokenizer = SentencePieceTokenizer()
sentence = "Hello, my name is John."
encoded = tokenizer.encode(sentence)
decoded = tokenizer.decode(encoded)
print(encoded)
print(decoded)  
print(tokenizer.get_vocab())
"""
