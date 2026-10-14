import torch
import torch.nn as nn
from torch.nn import functional as F

# Indstillinger (Skal matche din Colab-træning præcis)
block_size = 64
device = 'cpu' 
n_embd = 1000
n_head = 6
n_layer = 6

# Ordbogen (Modellen skal kende de samme tegn som i Colab)
# Udskift det gamle ordbogs-afsnit i VS Code med dette:
import tiktoken
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)
vocab_size = enc.n_vocab


# --- MODEL ARKITEKTUR ---
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
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
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

# --- INDLÆS DEN GEMTE MODEL ---
model = MiniCodeGPT().to(device)
model.load_state_dict(torch.load('mini_code_gpt.pth', map_location=device))
model.eval()

# =======================================================
# LOKAL CHAT-LØSNING (Uendeligt loop i terminalen)
# =======================================================
print("====================================================")
print("   VELKOMMEN TIL DIN PRIVATE KODE-AI CHAT-BOT       ")
print("   Skriv starten på noget kode (f.eks. 'def ')       ")
print("   Skriv 'exit' for at lukke chatten.               ")
print("====================================================\n")

while True:
    bruger_input = input("Du: ")
    
    if bruger_input.lower() == 'exit':
        print("\nFarvel! Chatten lukker.")
        break
        
    if not bruger_input.strip():
        continue

    print("\nAI tænker...", end="\r")

    try:
        context = torch.tensor([encode(bruger_input)], dtype=torch.long, device=device)
        genereret_output = model.generate(context, max_new_tokens=150).tolist()[0] # [0] henter den flade liste med det samme
        
        print("AI:")
        print(decode(genereret_output))
        print("-" * 50 + "\n")
        
    except KeyError:
        print("AI: Fejl! Du brugte et tegn eller bogstav, som jeg ikke lærte i min træning.\n")
