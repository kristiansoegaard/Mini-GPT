import os
import re
import sys
import time
import ctypes
import threading
import tkinter as tk
from tkinter import messagebox, filedialog, font as tkfont
from tkinter import ttk

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

MODEL_FILES = ('mini_chat_gpt.pth', 'mini_code_gpt.pth')

# ---------------- COLORS ----------------
BG = '#0d1020'
PANEL = '#151a30'
BORDER = '#262d4d'
TEXT = '#e6e9f7'
MUTED = '#8a91b4'
ACCENT = '#7c5cff'
ACCENT_HOVER = '#9378ff'
ACCENT2 = '#22d3ee'
CODE_BG = '#0a0d1a'
GREEN = '#34d399'
AMBER = '#fbbf24'
RED = '#f87171'
DISABLED = '#2a3050'

STATE_COLORS = {'ready': GREEN, 'busy': AMBER, 'loading': ACCENT2, 'error': RED}


def app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(name):
    return os.path.join(getattr(sys, '_MEIPASS', app_dir()), name)


# ---------------- MODEL (samme opbygning som din traeningskode) ----------------
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.scale = n_embd ** -0.5  # som i din oprindelige train_ai.py

    def forward(self, x):
        B, T, C = x.shape
        wei = self.query(x) @ self.key(x).transpose(-2, -1) * self.scale
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        return F.softmax(wei, dim=-1) @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, head_size, block_size):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class MiniCodeGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, head_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, head_size, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=idx.device))
        return self.lm_head(self.ln_f(self.blocks(x)))

    @torch.no_grad()
    def stream(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        """Giver ét nyt token ad gangen, så appen kan vise svaret live."""
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.block_size:])[:, -1, :] / max(temperature, 1e-5)
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat((idx, nxt), dim=1)
            yield nxt.item()


def load_model(path):
    sd = torch.load(path, map_location='cpu', weights_only=True)
    vocab_size, n_embd = sd['token_embedding_table.weight'].shape
    block_size = sd['position_embedding_table.weight'].shape[0]
    n_layer = len({k.split('.')[1] for k in sd if k.startswith('blocks.')})
    n_head = len({k.split('.')[4] for k in sd if k.startswith('blocks.0.sa.heads.')})
    head_size = sd['blocks.0.sa.heads.0.key.weight'].shape[0]
    model = MiniCodeGPT(vocab_size, block_size, n_embd, n_head, n_layer, head_size)
    model.load_state_dict(sd)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    info = f'{params:.1f}M params  ·  {n_embd} dim  ·  {n_layer} layers  ·  {n_head} heads  ·  context {block_size}'
    return model, info


def find_model_file():
    for folder in (app_dir(), os.getcwd()):
        for name in MODEL_FILES:
            p = os.path.join(folder, name)
            if os.path.exists(p):
                return p
    return filedialog.askopenfilename(title='Select your model (.pth)',
                                      filetypes=[('PyTorch model', '*.pth'), ('All files', '*.*')])


# ---------------- DRAWING HELPERS ----------------
def lerp(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return '#%02x%02x%02x' % tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def rrect(c, x1, y1, x2, y2, r, **kw):
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
           x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return c.create_polygon(pts, smooth=True, **kw)


def draw_logo(c, x, y, s=44):
    rrect(c, x, y, x + s, y + s, 12, fill=ACCENT, outline=lerp(ACCENT, ACCENT2, 0.5), tags='static')
    left = [(0.26, 0.28), (0.26, 0.50), (0.26, 0.72)]
    mid = [(0.50, 0.38), (0.50, 0.62)]
    right = [(0.76, 0.50)]
    P = lambda p: (x + p[0] * s, y + p[1] * s)
    for a in left:
        for b in mid:
            c.create_line(*P(a), *P(b), fill='#d9d0ff', width=1.4, tags='static')
    for a in mid:
        for b in right:
            c.create_line(*P(a), *P(b), fill='#d9d0ff', width=1.4, tags='static')
    for p in left + mid + right:
        px, py = P(p)
        r = s * 0.065
        c.create_oval(px - r, py - r, px + r, py + r, fill='white', outline='', tags='static')


class Btn(tk.Canvas):
    def __init__(self, parent, text, command, w=140, h=42, color=ACCENT, hover=ACCENT_HOVER,
                 fg='white', outline=None, font=None):
        super().__init__(parent, width=w, height=h, bg=parent.cget('bg'), highlightthickness=0, cursor='hand2')
        self.w, self.h, self.cmd, self.text = w, h, command, text
        self.color, self.hover, self.fg, self.outline, self.font = color, hover, fg, outline, font
        self.enabled = True
        self._render(self.color)
        self.bind('<Enter>', lambda e: self._render(self.hover) if self.enabled else None)
        self.bind('<Leave>', lambda e: self._render(self.color if self.enabled else DISABLED))
        self.bind('<Button-1>', self._click)

    def _render(self, fill):
        self.delete('all')
        rrect(self, 1, 1, self.w - 1, self.h - 1, 12, fill=fill, outline=self.outline or fill)
        self.create_text(self.w / 2, self.h / 2, text=self.text, fill=self.fg if self.enabled else MUTED, font=self.font)

    def _click(self, e):
        if self.enabled and self.cmd:
            self.cmd()

    def set_enabled(self, on):
        self.enabled = on
        self.config(cursor='hand2' if on else 'arrow')
        self._render(self.color if on else DISABLED)

    def set_text(self, text):
        self.text = text
        self._render(self.color if self.enabled else DISABLED)

    def set_colors(self, color, hover, fg, outline):
        self.color, self.hover, self.fg, self.outline = color, hover, fg, outline
        self._render(self.color if self.enabled else DISABLED)


# ---------------- SYNTAX COLORS ----------------
HIGHLIGHT = [
    ('number', r'\b\d+(?:\.\d+)?\b', '#ff9f6b'),
    ('builtin', r'\b(?:print|range|len|int|str|float|list|dict|set|sum|min|max|abs|input|open)\b', '#5ad1ff'),
    ('keyword', r'\b(?:def|return|if|elif|else|for|while|in|import|from|class|True|False|None|and|or|not|'
                r'try|except|with|as|pass|break|continue|lambda|is|yield|raise)\b', '#c792ff'),
    ('func', r'(?<=def )\w+', '#ffd479'),
    ('string', r'"[^"\n]*"|\'[^\'\n]*\'', '#8bdc8f'),
    ('comment', r'#[^\n]*', '#5c6486'),
]


class Slider(ttk.Scale):
    """Slider der runder til en fast step og viser vaerdien i en label."""
    def __init__(self, parent, lo, hi, step, value, label):
        self.step, self.label = step, label
        super().__init__(parent, from_=lo, to=hi, orient='horizontal', length=190,
                         style='Dark.Horizontal.TScale', command=self._changed)
        self.set(value)

    def value(self):
        v = round(float(super().get()) / self.step) * self.step
        return int(v) if self.step >= 1 else round(v, 1)

    def _changed(self, _=None):
        self.label.config(text=str(self.value()))


class App:
    def __init__(self, root, model_path):
        self.root = root
        self.model = None
        self.enc = None
        self.anim = None
        self.dots = 0
        self.state = 'loading'
        self.base_text = ''
        self.has_output = False
        self.stop_evt = threading.Event()
        self.history = []
        self.mode = 'chat' if 'chat' in os.path.basename(model_path).lower() else 'code'

        families = set(tkfont.families())
        self.ui = next((f for f in ('Segoe UI', 'Inter', 'Helvetica', 'DejaVu Sans') if f in families), 'TkDefaultFont')
        self.mono = next((f for f in ('Cascadia Code', 'Consolas', 'DejaVu Sans Mono', 'Courier New') if f in families),
                         'TkFixedFont')

        root.title('Mini-GPT')
        root.geometry('920x780')
        root.minsize(780, 660)
        root.configure(bg=BG)
        try:
            root.iconbitmap(resource_path('mini-gpt.ico'))
        except Exception:
            pass

        self._style()
        self._build_header()
        self._build_statusbar()
        self._build_body()

        root.bind('<Control-Return>', lambda e: self.start())
        root.bind('<Escape>', lambda e: self.stop())

        self.set_status('Loading model', 'loading')
        threading.Thread(target=self._load, args=(model_path,), daemon=True).start()

    # ----- styling & layout -----
    def _style(self):
        st = ttk.Style()
        st.theme_use('clam')
        st.configure('Dark.Vertical.TScrollbar', background=BORDER, troughcolor=CODE_BG, bordercolor=CODE_BG,
                     arrowcolor=MUTED, lightcolor=BORDER, darkcolor=BORDER, relief='flat')
        st.map('Dark.Vertical.TScrollbar', background=[('active', ACCENT)])
        st.configure('Dark.Horizontal.TScale', background=ACCENT, troughcolor=PANEL, bordercolor=BORDER,
                     lightcolor=ACCENT, darkcolor=ACCENT, gripcount=0)
        st.map('Dark.Horizontal.TScale', background=[('active', ACCENT_HOVER)])

    def _build_header(self):
        self.header = tk.Canvas(self.root, height=84, bg=BG, highlightthickness=0)
        self.header.pack(fill='x')
        draw_logo(self.header, 24, 20, 44)
        self.header.create_text(84, 34, anchor='w', text='Mini-GPT', fill=TEXT, font=(self.ui, 20, 'bold'))
        self.header.create_text(85, 58, anchor='w', text='Your private AI assistant', fill=MUTED,
                                font=(self.ui, 10))
        self.header.bind('<Configure>', self._draw_header)

    def _draw_header(self, e=None):
        c = self.header
        c.delete('dyn')
        w = c.winfo_width()
        n = max(1, w // 4)
        for i in range(n + 1):
            c.create_rectangle(i * 4, 81, i * 4 + 4, 84, fill=lerp(ACCENT, ACCENT2, i / n), width=0, tags='dyn')
        rrect(c, w - 216, 28, w - 24, 58, 15, fill=PANEL, outline=BORDER, tags='dyn')
        c.create_oval(w - 202, 38, w - 192, 48, fill=GREEN, outline='', tags='dyn')
        c.create_text(w - 182, 43, anchor='w', text='Runs locally  ·  offline', fill=MUTED,
                      font=(self.ui, 9), tags='dyn')

    def _label(self, parent, text):
        return tk.Label(parent, text=text, bg=BG, fg=MUTED, font=(self.ui, 8, 'bold'))

    def _build_body(self):
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill='both', expand=True, padx=24, pady=(16, 0))

        # mode switch
        modes = tk.Frame(body, bg=BG)
        modes.pack(fill='x', pady=(0, 12))
        self.btn_chat = Btn(modes, 'Chat', lambda: self.set_mode('chat'), w=90, h=32, font=(self.ui, 10, 'bold'))
        self.btn_chat.pack(side='left')
        self.btn_code = Btn(modes, 'Code', lambda: self.set_mode('code'), w=90, h=32, font=(self.ui, 10, 'bold'))
        self.btn_code.pack(side='left', padx=(8, 0))

        # prompt
        self.prompt_label = self._label(body, 'PROMPT')
        self.prompt_label.pack(anchor='w')
        card = tk.Frame(body, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        card.pack(fill='x', pady=(6, 14))
        self.prompt = tk.Text(card, height=4, bg=PANEL, fg=TEXT, insertbackground=TEXT, relief='flat',
                              padx=14, pady=12, font=(self.mono, 12), wrap='word', undo=True,
                              selectbackground=ACCENT)
        self.prompt.pack(fill='x')
        self.prompt.insert('1.0', 'def ' if self.mode == 'code' else '')
        self.prompt.bind('<FocusIn>', lambda e: card.config(highlightbackground=ACCENT))
        self.prompt.bind('<FocusOut>', lambda e: card.config(highlightbackground=BORDER))
        self.prompt.bind('<Control-Return>', lambda e: (self.start(), 'break')[1])
        self.prompt.bind('<Return>', self._on_return)

        # controls
        row = tk.Frame(body, bg=BG)
        row.pack(fill='x', pady=(0, 16))
        self.length = self._slider(row, 'LENGTH (TOKENS)', 10, 500, 10, 150)
        self.temp = self._slider(row, 'CREATIVITY', 0.1, 1.5, 0.1, 0.8)
        self.btn_gen = Btn(row, 'Generate  ▶', self.start, w=150, h=44, font=(self.ui, 11, 'bold'))
        self.btn_gen.pack(side='right')
        self.btn_stop = Btn(row, 'Stop', self.stop, w=90, h=44, color=BG, hover=PANEL, outline=BORDER,
                            fg=TEXT, font=(self.ui, 10, 'bold'))
        self.btn_stop.pack(side='right', padx=(0, 10))
        self.btn_stop.set_enabled(False)
        self.btn_gen.set_enabled(False)

        # output header
        head = tk.Frame(body, bg=BG)
        head.pack(fill='x')
        self._label(head, 'OUTPUT').pack(side='left')
        small = (self.ui, 9)
        Btn(head, 'Clear', self.clear, w=70, h=28, color=BG, hover=PANEL, outline=BORDER, fg=MUTED,
            font=small).pack(side='right')
        Btn(head, 'Copy', self.copy, w=70, h=28, color=BG, hover=PANEL, outline=BORDER, fg=MUTED,
            font=small).pack(side='right', padx=(0, 8))

        # output
        ocard = tk.Frame(body, bg=CODE_BG, highlightthickness=1, highlightbackground=BORDER)
        ocard.pack(fill='both', expand=True, pady=(6, 14))
        sb = ttk.Scrollbar(ocard, style='Dark.Vertical.TScrollbar')
        sb.pack(side='right', fill='y')
        self.out = tk.Text(ocard, bg=CODE_BG, fg='#d6dbf5', relief='flat', padx=16, pady=14,
                           font=(self.mono, 12), wrap='word', yscrollcommand=sb.set,
                           selectbackground=ACCENT, spacing1=2, spacing3=2)
        self.out.pack(side='left', fill='both', expand=True)
        sb.config(command=self.out.yview)
        self.out.tag_configure('placeholder', foreground=MUTED, font=(self.mono, 11, 'italic'))
        for tag, _, color in HIGHLIGHT:
            self.out.tag_configure(tag, foreground=color)
        self.out.tag_configure('you_label', foreground=ACCENT2, font=(self.ui, 9, 'bold'))
        self.out.tag_configure('you', foreground=TEXT, font=(self.ui, 12))
        self.out.tag_configure('ai_label', foreground='#b39dff', font=(self.ui, 9, 'bold'))
        self.out.tag_configure('ai', foreground='#d6dbf5', font=(self.ui, 12))
        self._apply_mode()

    def _slider(self, parent, title, lo, hi, step, value):
        box = tk.Frame(parent, bg=BG)
        box.pack(side='left', padx=(0, 26))
        top = tk.Frame(box, bg=BG)
        top.pack(fill='x')
        self._label(top, title).pack(side='left')
        val = tk.Label(top, bg=BG, fg=ACCENT2, font=(self.ui, 9, 'bold'))
        val.pack(side='right')
        slider = Slider(box, lo, hi, step, value, val)
        slider.pack(pady=(6, 0))
        return slider

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=PANEL)
        bar.pack(fill='x', side='bottom')
        self.dot = tk.Canvas(bar, width=14, height=14, bg=PANEL, highlightthickness=0)
        self.dot.pack(side='left', padx=(18, 6), pady=10)
        self.dot_id = self.dot.create_oval(2, 2, 12, 12, fill=ACCENT2, outline='')
        self.status = tk.Label(bar, text='', bg=PANEL, fg=TEXT, font=(self.ui, 9))
        self.status.pack(side='left')
        self.info = tk.Label(bar, text='', bg=PANEL, fg=MUTED, font=(self.ui, 9))
        self.info.pack(side='right', padx=18)

    # ----- status -----
    def set_status(self, text, state='ready'):
        if self.anim:
            self.root.after_cancel(self.anim)
            self.anim = None
        self.state, self.base_text = state, text
        self.status.config(text=text)
        self.dot.itemconfig(self.dot_id, fill=STATE_COLORS[state])
        if state in ('busy', 'loading'):
            self.dots = 0
            self._tick()

    def _tick(self):
        if self.state not in ('busy', 'loading'):
            return
        self.dots = (self.dots + 1) % 4
        self.status.config(text=self.base_text + '.' * self.dots)
        self.anim = self.root.after(350, self._tick)

    # ----- model loading -----
    def _load(self, path):
        try:
            self.model, info = load_model(path)
            self.enc = tiktoken.get_encoding('gpt2')
            self.root.after(0, self._loaded, info)
        except Exception as e:
            self.root.after(0, self._load_failed, str(e))

    def _loaded(self, info):
        self.info.config(text=info)
        self.btn_gen.set_enabled(True)
        self.set_status(self.ready_text(), 'ready')

    def _load_failed(self, err):
        self.set_status('Could not load model', 'error')
        messagebox.showerror('Could not start', f'{err}\n\nMake sure {MODEL_FILES[0]} is next to the program, '
                                                'and that you have internet the first time '
                                                '(the tokenizer has to be downloaded).')
        self.root.destroy()

    # ----- generating -----
    def start(self):
        if self.model is None or not self.btn_gen.enabled:
            return
        raw = self.prompt.get('1.0', 'end-1c')
        text = raw.strip() if self.mode == 'chat' else raw
        if not text.strip():
            return
        n = int(self.length.value())
        temp = float(self.temp.value())
        self.stop_evt.clear()
        self.btn_gen.set_enabled(False)
        self.btn_stop.set_enabled(True)
        self.set_status('Thinking' if self.mode == 'chat' else 'Generating', 'busy')
        if self.mode == 'chat':
            self.prompt.delete('1.0', 'end')
            threading.Thread(target=self._run_chat, args=(text, self._chat_prompt(text), n, temp), daemon=True).start()
        else:
            threading.Thread(target=self._run, args=(text, n, temp), daemon=True).start()

    def stop(self):
        if self.btn_stop.enabled:
            self.stop_evt.set()

    def _run(self, text, n, temp):
        t0 = time.time()
        count = 0
        try:
            ids = self.enc.encode(text, disallowed_special=())
            ctx = torch.tensor([ids], dtype=torch.long)
            self.root.after(0, self.show, text)
            for tok in self.model.stream(ctx, n, temp, 40):
                if self.stop_evt.is_set() or tok == self.enc.eot_token:
                    break
                ids.append(tok)
                count += 1
                self.root.after(0, self.show, self.enc.decode(ids))
            self.root.after(0, self._finish, count, time.time() - t0, self.stop_evt.is_set(), None)
        except Exception as e:
            self.root.after(0, self._finish, 0, 0, False, str(e))

    def _finish(self, count, secs, stopped, error):
        self.btn_gen.set_enabled(True)
        self.btn_stop.set_enabled(False)
        if error:
            self.set_status('Something went wrong', 'error')
            messagebox.showerror('Error', error)
            return
        rate = count / secs if secs > 0 else 0
        word = 'Stopped' if stopped else 'Done'
        self.set_status(f'{word}  ·  {count} tokens  ·  {rate:.0f} tokens/sec', 'ready')

    # ----- output -----
    def show(self, text):
        self.has_output = True
        o = self.out
        o.config(state='normal')
        o.delete('1.0', 'end')
        o.insert('1.0', text)
        for tag, pattern, _ in HIGHLIGHT:
            for m in re.finditer(pattern, text):
                o.tag_add(tag, f'1.0+{m.start()}c', f'1.0+{m.end()}c')
        o.see('end')
        o.config(state='disabled')

    def clear(self):
        self.history = []
        self._refresh_output()

    def _refresh_output(self):
        if self.mode == 'chat' and self.history:
            self.show_chat()
            return
        self.has_output = False
        o = self.out
        o.config(state='normal')
        o.delete('1.0', 'end')
        msg = ('Say hi to start chatting.\nPress Enter to send, Shift+Enter for a new line.'
               if self.mode == 'chat' else
               'Your generated code will appear here.\nTip: press Ctrl+Enter to generate.')
        o.insert('1.0', msg, 'placeholder')
        o.config(state='disabled')

    # ----- chat / code mode -----
    def ready_text(self):
        return ('Ready  ·  press Enter to send' if self.mode == 'chat'
                else 'Ready  ·  press Ctrl+Enter to generate')

    def _on_return(self, e):
        if self.mode == 'chat' and not (e.state & 0x1):  # Shift+Enter = new line
            self.start()
            return 'break'

    def _apply_mode(self):
        chat = self.mode == 'chat'
        on = dict(color=ACCENT, hover=ACCENT_HOVER, fg='white', outline=None)
        off = dict(color=BG, hover=PANEL, fg=MUTED, outline=BORDER)
        self.btn_chat.set_colors(**(on if chat else off))
        self.btn_code.set_colors(**(off if chat else on))
        self.prompt_label.config(text='MESSAGE' if chat else 'PROMPT')
        self.btn_gen.set_text('Send  ▶' if chat else 'Generate  ▶')
        self._refresh_output()

    def set_mode(self, mode):
        if mode == self.mode or self.btn_stop.enabled:
            return
        if self.prompt.get('1.0', 'end-1c').strip() in ('', 'def'):
            self.prompt.delete('1.0', 'end')
            if mode == 'code':
                self.prompt.insert('1.0', 'def ')
        self.mode = mode
        self._apply_mode()
        if self.model is not None:
            self.set_status(self.ready_text(), 'ready')

    def _chat_prompt(self, text):
        past = ''.join(f'User: {u}\nAI: {a}\n' for u, a in self.history[-3:])
        return past + f'User: {text}\nAI:'

    def show_chat(self, pending_user=None, reply=''):
        o = self.out
        o.config(state='normal')
        o.delete('1.0', 'end')
        turns = list(self.history)
        if pending_user is not None:
            turns.append((pending_user, reply))
        for u, a in turns:
            o.insert('end', 'YOU\n', 'you_label')
            o.insert('end', u + '\n\n', 'you')
            o.insert('end', 'AI\n', 'ai_label')
            o.insert('end', (a or '…') + '\n\n', 'ai')
        self.has_output = bool(turns)
        o.see('end')
        o.config(state='disabled')

    def _run_chat(self, user_text, prompt, n, temp):
        t0 = time.time()
        count = 0
        reply_ids = []
        reply = ''
        try:
            ids = self.enc.encode(prompt, disallowed_special=())
            ctx = torch.tensor([ids], dtype=torch.long)
            self.root.after(0, self.show_chat, user_text, '')
            for tok in self.model.stream(ctx, n, temp, 40):
                if self.stop_evt.is_set() or tok == self.enc.eot_token:
                    break
                reply_ids.append(tok)
                count += 1
                reply = self.enc.decode(reply_ids)
                cuts = [i for i in (reply.find('\nUser:'), reply.find('\nAI:')) if i != -1]
                if cuts:
                    reply = reply[:min(cuts)]
                    break
                self.root.after(0, self.show_chat, user_text, reply.strip())
            self.root.after(0, self._chat_done, user_text, reply.strip(), count, time.time() - t0,
                            self.stop_evt.is_set())
        except Exception as e:
            self.root.after(0, self._finish, 0, 0, False, str(e))

    def _chat_done(self, user_text, reply, count, secs, stopped):
        self.history.append((user_text, reply or '…'))
        self.show_chat()
        self._finish(count, secs, stopped, None)
        self.prompt.focus_set()

    def copy(self):
        if not self.has_output:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.out.get('1.0', 'end-1c'))
        old = self.base_text
        self.status.config(text='Copied to clipboard')
        self.root.after(1500, lambda: self.status.config(text=old) if self.state == 'ready' else None)


def main():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # skarp tekst paa Windows
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('mini-gpt.app')
    except Exception:
        pass
    root = tk.Tk()
    root.withdraw()
    path = find_model_file()
    if not path:
        root.destroy()
        return
    root.deiconify()
    App(root, path)
    root.mainloop()


if __name__ == '__main__':
    main()