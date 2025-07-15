# dependencies
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameter:

batch_size = 64 # how many independend sequences we will be processing parallely
block_size = 256 # max context length
max_iters = 5000 # total no's of training steps
eval_interval = 500 # eval modal after 500 steps
learning_rate = 3e-4 # lr for the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # no of mini-batchs to avg over during evals
n_embd = 384 # size of embedding tokens
n_head = 6 # num of attn-heads
n_layer = 6 # num of transformer block
dropout = 0.2 # dp probs

# Data Loading and Encoding:

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    # Loads shakespeare text from file
    text = f.read()
    
chars = sorted(list(set(text))) # finds the set of all unique chars
vocab_size = len(chars) # builds vocab

# building a lookup table
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}

# encoder & decoder builder -> chars to int64 mapping
encode = lambda s : {stoi[c] for c in s}
decode = lambda l : ''.join([itos[i] for i in l])

# data Loader -> splits the encoded data into training (90%) and validation (10%):
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Batch Generations:
def get_batch(split):
    
    """
    > Randomly picks *BATCH_SIZE* sequences of length *BLOCK_SIZE* from either train or val data
    > Returns i/p x and next-token target y
    """
    
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size+1] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits. loss = model(X, Y)
            losses[k] = loss.item()
        
        out[split] = losses.mean()
    model.train()
    return out


    
    