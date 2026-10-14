"""
Trains Mini-GPT to CHAT (and optionally still write code).

Setup (once):
    pip install torch tiktoken datasets

Data it uses (all optional, but you need at least one chat source):
  1. A small chat dataset downloaded automatically from Hugging Face
     (HuggingFaceTB/everyday-conversations-llama3.1-2k, ~2,200 short conversations).
  2. chat.txt   - your own conversations in this format, separated by a blank line:
         User: hi
         AI: Hello! How can I help?
         User: what is 2+2?
         AI: 2 + 2 is 4.

         User: next conversation...
  3. dataset.txt - your code file. It is mixed in so the model keeps some code skills.

Output: mini_chat_gpt.pth  (the app loads it automatically, in Chat mode).
Run it on a GPU (e.g. Google Colab) - on a plain CPU it will be very slow.
"""
import os
import re
import math
import random
import contextlib
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# ---------------- SETTINGS ----------------
block_size = 256        # how much text the model can "see" at once
n_embd = 256
n_head = 8
n_layer = 6
dropout = 0.2           # helps against memorizing the small dataset
batch_size = 32
max_iters = 3000
eval_interval = 250
eval_iters = 20
learning_rate = 3e-4
min_lr = 3e-5
warmup_iters = 100
val_fraction = 0.05

SAVE_AS = 'mini_chat_gpt.pth'
CHAT_DATASET = 'HuggingFaceTB/everyday-conversations-llama3.1-2k'
LOCAL_CHAT_FILE = 'chat.txt'
LOCAL_CODE_FILE = 'dataset.txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)
random.seed(1337)
print(f'Using hardware: {device}')
if device == 'cpu':
    print('[WARNING] No GPU found - training will be slow. Use Google Colab with a GPU if you can.')

enc = tiktoken.get_encoding('gpt2')
vocab_size = enc.n_vocab
eot = enc.eot_token  # <|endoftext|> marks "the conversation is over"


# ---------------- DATA ----------------
def to_text(messages):
    lines = []
    for m in messages:
        role = m.get('role', m.get('from'))
        content = str(m.get('content', m.get('value', ''))).strip()
        if not content or role == 'system':
            continue
        name = 'User' if role in ('user', 'human') else 'AI'
        lines.append(f'{name}: {content}')
    return '\n'.join(lines) + '\n'


def load_hf_chats():
    try:
        from datasets import load_dataset
    except ImportError:
        print('[INFO] "datasets" is not installed. Run: pip install datasets')
        return []
    try:
        ds = load_dataset(CHAT_DATASET)
    except Exception as e:
        print(f'[WARNING] Could not download {CHAT_DATASET}: {e}')
        return []
    chats = []
    for split in ds.values():
        for ex in split:
            msgs = ex.get('messages') or ex.get('conversations')
            if not msgs:
                msgs = next((v for v in ex.values()
                             if isinstance(v, list) and v and isinstance(v[0], dict)), None)
            if msgs:
                chats.append(to_text(msgs))
    print(f'[SUCCESS] Loaded {len(chats)} conversations from {CHAT_DATASET}')
    return chats


def load_local_chats():
    if not os.path.exists(LOCAL_CHAT_FILE):
        return []
    with open(LOCAL_CHAT_FILE, 'r', encoding='utf-8') as f:
        raw = f.read()
    chats = [c.strip() + '\n' for c in re.split(r'\n\s*\n', raw) if c.strip()]
    print(f'[SUCCESS] Loaded {len(chats)} conversations from {LOCAL_CHAT_FILE}')
    return chats


chats = load_hf_chats() + load_local_chats()
if not chats:
    raise SystemExit('No chat data found. Install "datasets" (pip install datasets) '
                     f'or create a {LOCAL_CHAT_FILE} file.')

random.shuffle(chats)
n_val = max(1, int(len(chats) * val_fraction))
val_chats, train_chats = chats[:n_val], chats[n_val:]


def tokenize(chat_list):
    ids = []
    for c in chat_list:
        ids += enc.encode_ordinary(c) + [eot]
    return ids


train_ids = tokenize(train_chats)
val_ids = tokenize(val_chats)

if os.path.exists(LOCAL_CODE_FILE):
    with open(LOCAL_CODE_FILE, 'r', encoding='utf-8') as f:
        code_ids = enc.encode_ordinary(f.read())[:len(train_ids)] + [eot]  # at most as much code as chat
    train_ids += code_ids
    print(f'[SUCCESS] Mixed in {len(code_ids)} code tokens from {LOCAL_CODE_FILE}')

train_data = torch.tensor(train_ids, dtype=torch.long)
val_data = torch.tensor(val_ids, dtype=torch.long)
assert len(val_data) > block_size + 1, 'Not enough validation data - add more conversations.'
print(f'Training tokens: {len(train_data):,}   Validation tokens: {len(val_data):,}')


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------------- MODEL ----------------
# Same layout as the app expects. Dropout has no weights, so saved files stay compatible.
# (Attention is scaled by n_embd**-0.5 on purpose, to match your original training code and the app.)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.drop = nn.Dropout(dropout)
        self.scale = n_embd ** -0.5

    def forward(self, x):
        B, T, C = x.shape
        wei = self.query(x) @ self.key(x).transpose(-2, -1) * self.scale
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = self.drop(F.softmax(wei, dim=-1))
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.net(x))


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class MiniCodeGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=idx.device))
        logits = self.lm_head(self.ln_f(self.blocks(self.drop(x))))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, vocab_size).float(), targets.view(B * T))
        return logits, loss


model = MiniCodeGPT().to(device)
print(f'Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

use_bf16 = device == 'cuda' and torch.cuda.is_bf16_supported()


def amp():
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    t = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ('train', 'val'):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            with amp():
                _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def chat_reply(message, max_new=80, temperature=0.7, top_k=40):
    model.eval()
    x = torch.tensor([enc.encode_ordinary(f'User: {message}\nAI:')], dtype=torch.long, device=device)
    out = []
    for _ in range(max_new):
        logits, _ = model(x[:, -block_size:])
        logits = logits[:, -1, :].float() / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float('-inf')
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        if nxt.item() == eot:
            break
        x = torch.cat((x, nxt), dim=1)
        out.append(nxt.item())
        if '\nUser:' in enc.decode(out):
            break
    return enc.decode(out).split('\nUser:')[0].strip()


# ---------------- TRAINING ----------------
best_val = float('inf')
print('Starting training...')
for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        l = estimate_loss()
        mark = ''
        if l['val'] < best_val:
            best_val = l['val']
            torch.save(model.state_dict(), SAVE_AS)
            mark = '  <- best so far, saved'
        print(f"Step {it}: train loss {l['train']:.4f}, val loss {l['val']:.4f}{mark}")
    for g in optimizer.param_groups:
        g['lr'] = get_lr(it)
    xb, yb = get_batch('train')
    with amp():
        _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

# ---------------- RESULT ----------------
model.load_state_dict(torch.load(SAVE_AS, map_location=device))
print(f'\n[SUCCESS] Saved the best model (val loss {best_val:.4f}) to {SAVE_AS}\n')
print('Quick test:')
for msg in ('hi', 'What is the sun?', 'Can you help me plan my weekend?'):
    print(f'  User: {msg}\n  AI:   {chat_reply(msg)}\n')