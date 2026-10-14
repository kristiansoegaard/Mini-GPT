import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparametre (Indstillinger for din AI)
batch_size = 32       # Hvor mange tekststykker vi træner på ad gangen
block_size = 64        # Hvor lang en kodekontekst AI'en kan kigge på
max_iters = 2000       # Hvor længe AI'en skal træne
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Brug GPU hvis muligt
n_embd = 1000       # Hvor mange "hjerneceller" (dimensioner) hvert tegn har
n_head = 6       # Antal opmærksomhedshoveder (Attention heads)
n_layer = 6         # Hvor dybt netværket er

print(f"Bruger hardware: {device}")

# 1. TRIN: TRÆNINGSDATA (Vi bruger et lille eksempel på Python-kode)
# I virkeligheden ville du loade en stor fil med tusindvis af linjers kode her.
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
""" * 50 # Vi forlænger teksten, så modellen har noget at øve sig på

# Find alle unikke tegn i koden (vores ordbog)
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Lav værktøjer til at oversætte tekst til tal og omvendt
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])

# Konverter hele vores datasæt til tal
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Funktion til at hente tilfældige bidder af kode til træning
def get_batch(split):
    data_set = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_set) - block_size, (batch_size,))
    x = torch.stack([data_set[i:i+block_size] for i in ix])
    y = torch.stack([data_set[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# 2. TRIN: SELVE AI-ARKITEKTUREN (Mini-Transformer)
class Head(nn.Module):
    """ Et enkelt Attention-hoved. Gør at tegn kan 'tale' med hinanden """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ Flere attention-hoveder, der kører parallelt """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """ Et simpelt lineært netværk, hvor modellen kan tænke over dataen """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ En transformer-blok: Kombinerer opmærksomhed og tænkning """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniCodeGPT(nn.Module):
    """ Det samlede neurale netværk """
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """ Generer ny kode baseret på et input """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# 3. TRIN: TRÆNING AF MODELLEN
model = MiniCodeGPT()
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Starter træning...")
for iter in range(max_iters):
    if iter % 500 == 0:
        xb, yb = get_batch('val')
        logits, loss = model(xb, yb)
        print(f"Trin {iter}: Fejlmargen (Loss) er {loss.item():.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 4. TRIN: TEST DIN AI!
print("\n--- TRÆNING FÆRDIG ---")
print("Vi beder nu AI'en om selv at fuldføre koden baseret på inputtet 'def ':\n")

context = torch.tensor([encode("def ")], dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=150)[0].tolist()))
# GEM MODELLEN TIL DISKEN
torch.save(model.state_dict(), 'mini_code_gpt.pth')
print("Modellens hjerne er gemt som 'mini_code_gpt.pth'!")
