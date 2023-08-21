# %%
import torch
import matplotlib.pyplot as plt
import wandb
import numpy as np

from utils.sentencepiece_tokenizer import SentencePieceTokenizer
from config import get_num_stories
from dataset import TinyStoriesDataset, pad
from model import DecoderModel


# %%

num_stories = get_num_stories()
wb = False
learning_rate = 0.001
max_seq_len = 2200
epochs = 100
batch_size = 16
embedding_dimensions = 32

# %%
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is: " + str(device))

torch.manual_seed(42)

tokenizer = SentencePieceTokenizer()
vocab_len = tokenizer.get_vocab_size()
dataset = TinyStoriesDataset(tokenizer)

if wb:
    wandb.init(
        project="tiny-stories-decoder",
        config={
            "learning_rate": learning_rate,
            "max_seq_len": max_seq_len,
            "vocab_len": vocab_len,
            "batch_size": batch_size,
            "dataset": dataset.get_data_filename(),
            "epochs": epochs,
            "architecture": "decoder-only"
        }
    )

num_train = int(0.8 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, len(dataset) - num_train])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad)


model = DecoderModel(max_seq_len, vocab_len, embedding_dimensions)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %%
for epoch in range(epochs):
    print("Epoch:", epoch)
    for idx, batch in enumerate(train_loader):
        sos = torch.full((batch.shape[0], 1), 1)
        eos = torch.full((batch.shape[0], 1), 2)

        x = batch
        x = torch.cat([sos, x], dim=1)
        y = torch.cat([x[:, 1:], eos], dim=1)

        x = x.to(device)
        y = y.to(device)

        probabilities = model(x)
        loss = torch.nn.functional.cross_entropy(
            probabilities.view(-1, vocab_len), y.view(-1), ignore_index=0)
        if wb:
            wandb.log({"loss": loss})
        if idx % 1000 == 0:
            print("Loss:", loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    val_losses = []
    for idx, batch in enumerate(val_loader):
        sos = torch.full((batch.shape[0], 1), 1)
        eos = torch.full((batch.shape[0], 1), 2)

        x_val = batch
        x_val = torch.cat([sos, x_val], dim=1) 
        y_val = torch.cat([x_val[:, 1:], eos], dim=1)
        
        x_val, y_val = x_val.to(device), y_val.to(device)   
        
        probabilities = model(x_val)
        loss = torch.nn.functional.cross_entropy(probabilities.view(-1, vocab_len), y_val.view(-1), ignore_index=0)
        if idx % 1000 == 0:
            print("Validation Loss:", loss.item())   
    
    if wb:
        wandb.finish()

# %%
print("Running generate...")
max_seq_len = 20


def generate_from_string(string):
    sos = torch.full((1, 1), 1)
    x = torch.cat([sos, torch.tensor([tokenizer.encode(string)])], dim=1)
    x = x.to(device)
    while True:
        p = model(x)
        p = torch.nn.functional.softmax(p[:, -1, :], dim=1)
        max_probability_response = torch.argmax(p, dim=1)
        max_probability_token = int(max_probability_response[0])
        p = max_probability_response.unsqueeze(0)
        x = torch.cat((x, p), dim=-1)
        if max_probability_token == 2 or len(x[0].tolist()) >= max_seq_len:
            break
    print("Generate:", tokenizer.decode(x[0].tolist()))
    print(x[0].tolist())


generate_from_string("The man")
generate_from_string("The woman")
generate_from_string("In the beginning")
generate_from_string("Once upon a time")
generate_from_string("Once upon a time, there was a")
