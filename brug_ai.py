import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# Hyperparameters (Must match your Colab training exactly)
block_size = 64
device = 'cpu' 
n_embd = 128
n_head = 4
n_layer = 4

# Dataset configuration & Tokenizer (Using GPT-2 vocabulary)
text = """
def is_even(number):
    if number % 2 == 0:
        return True
    else:
        return False

def calculate_square(n):
    result = n * n
    return result

for i in range(10):
    print(is_even(i))
""" * 50

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)
vocab_size = enc.n_vocab

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
        k = self.key(x); q = self.query(x)
        wei = q @ k.transpose(-2,-1) * x.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ self.value(x)

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
    def forward(self, idx):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=device))
        return self.lm_head(self.ln_f(self.blocks(x))), None
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx

# --- LOAD TRAINED WEIGHTS ---
model = MiniCodeGPT().to(device)
model.load_state_dict(torch.load('mini_code_gpt.pth', map_location=device))
model.eval()

# =======================================================
# LOCAL INTERACTIVE CHAT-LOOP
# =======================================================
print("====================================================")
print("   WELCOME TO YOUR PRIVATE CODE-GPT CHATBOT        ")
print("   Provide a code prompt (e.g., 'def ')            ")
print("   Type 'exit' to close the session.               ")
print("====================================================\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'exit':
        print("\nGoodbye! Session closed.")
        break
        
    if not user_input.strip():
        continue

    print("\nAI is thinking...", end="\r")

    try:
        context = torch.tensor([encode(user_input)], dtype=torch.long, device=device)
        generated_output = model.generate(context, max_new_tokens=150).tolist()
        
        print("AI:")
        print(decode(generated_output[0])) # optimized to access flat list safely
        print("-" * 50 + "\n")
        
    except KeyError:
        print("AI: Error! You used a character that was not included in my training data.\n")
