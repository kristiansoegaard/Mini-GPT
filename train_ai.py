import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import tiktoken

# 1. HYPERPARAMETERS
batch_size = 32       
block_size = 64        
max_iters = 2000       
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
n_embd = 128           
n_head = 4             
n_layer = 4            

print(f"Using hardware: {device}")

# 2. DATASET LOADING (Bypassing Python network layers using wget)
print("Downloading training data directly to local disk...")
url = "https://githubusercontent.com"
if not os.path.exists('dataset.txt'):
    os.system(f"wget {url} -O dataset.txt")

try:
    with open('dataset.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"[SUCCESS] Loaded {len(text)} characters from dataset!")
except Exception as e:
    print(f"Fallback to dummy data. Error: {e}")
    text = "def is_even(n):\n    return n % 2 == 0\n" * 500

# 3. TOKENIZATION (Upgraded to professional TikToken vocabulary)
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)
vocab_size = enc.n_vocab

# Split dataset into training and validation
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_set = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_set) - block_size, (batch_size,))
    x = torch.stack([data_set[i:i+block_size] for i in ix])
    y = torch.stack([data_set[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# --- MODEL ARCHITECTURE (Transformer) ---
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B,T,C = x.shape
        wei = self.query(x) @ self.key(x).transpose(-2,-1) * x.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        return F.softmax(wei, dim=-1) @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
    def forward(self, x): return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd); self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))

class MiniCodeGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=device))
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        loss = F.cross_entropy(logits.view(B*T, vocab_size), targets.view(B*T)) if targets is not None else None
        return logits, loss

# 4. TRAINING LOOP
model = MiniCodeGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Starting local fallback training...")
for iter in range(max_iters):
    if iter % 500 == 0:
        xb, yb = get_batch('val')
        _, loss = model(xb, yb)
        print(f"Step {iter}: Validation Loss is {loss.item():.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 5. SAVE MODEL WEIGHTS
print("Saving model architecture...")
torch.save(model.state_dict(), 'mini_code_gpt.pth')
print("[SUCCESS] Saved weights to 'mini_code_gpt.pth'")
