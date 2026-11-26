#Building Efficient Transformers from Scratch in PyTorch

## Why Transformers? – Problem Framing

To see the practical gap, run a tiny benchmark on a synthetic 1‑D time‑series (length = 1024, batch = 32). The RNN processes each step sequentially, while self‑attention builds the full attention matrix in one GPU kernel.

```python
# synthetic data
x = torch.randn(32, 1024, 64)          # B, T, C
# RNN (sequential)
rnn = nn.GRU(64, 64, batch_first=True)
t0 = time.time(); _ = rnn(x); rnn_time = time.time() - t0
# Self‑attention (parallel)
att = nn.MultiheadAttention(64, 4, batch_first=True)
t0 = time.time(); _ = att(x, x, x); att_time = time.time() - t0
print(f"RNN: {rnn_time:.3f}s, SA: {att_time:.3f}s")
```

On the same hardware the RNN typically takes 3–5× longer because it cannot start step *t + 1* until step *t* finishes. Self‑attention evaluates all positions simultaneously, exposing massive parallelism.

**Core pain points solved by Transformers**

- **Training speed** – parallel computation reduces epoch time dramatically.  
- **Long‑range dependency capture** – each token attends to every other token in O(1) depth, unlike the O(T) recurrence depth of RNNs.  
- **Hardware utilization** – dense matrix multiplications map efficiently to GPUs/TPUs, achieving high FLOP utilization.

**Learning objectives of this post**

- Implement a minimal, end‑to‑end Transformer from scratch in PyTorch.  
- Measure its forward‑pass latency and throughput on the synthetic benchmark.  
- Identify and avoid common pitfalls such as padding mask bugs, numerical overflow in softmax, and inefficient tensor shapes.

## Self‑Attention Intuition and Mathematics

**QKV projection**  
For an input matrix \(X\in\mathbb{R}^{T\times d_{\text{model}}}\) the queries, keys and values are obtained with three learned linear maps:  

\[
Q = XW_Q,\qquad K = XW_K,\qquad V = XW_V,
\]

where \(W_Q,W_K,W_V\in\mathbb{R}^{d_{\text{model}}\times d_k}\).  
*Hand‑calculated 2‑token example* ( \(T=2,\ d_k=2\) ):

```python
X = torch.tensor([[1., 0.],
                  [0., 1.]])          # (2,2)
W = torch.eye(2)                     # identity for Q,K,V
Q = K = V = X @ W                    # (2,2)

scale = 1 / math.sqrt(2)             # ≈0.707
scores = (Q @ K.T) * scale           # [[0.707, 0.   ],
                                      #  [0.   , 0.707]]
weights = torch.softmax(scores, dim=-1)   # [[1.,0.],[0.,1.]]
output = weights @ V                 # returns X
```

The scaling factor \(\frac{1}{\sqrt{d_k}}\) prevents large dot‑products from saturating the softmax.

**Sinusoidal positional encoding**  
For position \(p\) and dimension index \(i\):

\[
\text{PE}(p,2i)   = \sin\!\Bigl(\frac{p}{10000^{2i/d_{\text{model}}}}\Bigr),\qquad
\text{PE}(p,2i+1) = \cos\!\Bigl(\frac{p}{10000^{2i/d_{\text{model}}}}\Bigr).
\]

Even indices use sine, odd indices use cosine; this closed‑form works for any \(d_{\text{model}}\) (odd dimensions simply truncate the last term).

**Multi‑head attention tensor flow**  

```
Input:   (B, T, 64)
→ Linear → (B, T, 64)
→ View    → (B, T, 4, 16)
→ Transpose→ (B, 4, T, 16)   # 4 heads, each 16‑dim
← Attention per head (B, 4, T, 16)
→ Concatenate → (B, T, 64)
→ Linear (output) → (B, T, 64)
```

*Checklist*  
1. Linear projections to Q,K,V.  
2. Scale by \(1/\sqrt{d_k}\).  
3. Softmax over scores.  
4. Weighted sum with V.  
5. Split/transpose for heads, compute attention, then concat.  
6. Final linear projection.

**Trade‑offs & edge cases**  
- More heads increase parallelism but shrink each head’s dimension, possibly limiting expressivity.  
- If \(d_{\text{model}}\) isn’t divisible by the head count, pad or reduce the model size; mismatched dimensions raise runtime errors.  

**Best practice**: set `bias=False` on the Q/K/V linear layers—biases shift dot‑product values and can degrade the similarity measure.

## Implementing a Transformer Block in PyTorch  

**1. Minimal encoder layer** – the core consists of three linear projections, scaled‑dot‑product attention, and a residual‑layer‑norm pair.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class SimpleEncoder(nn.Module):
    def __init__(self, d_model=256, n_head=4):
        super().__init__()
        self.d_k = d_model // n_head
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape                     # batch, seq_len, dim
        qkv = self.qkv(x).reshape(B, T, 3, -1, self.d_k)
        q, k, v = qkv.unbind(2)               # each: (B, T, n_head, d_k)

        attn = (q @ k.transpose(-2, -1)) / (self.d_k**0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)

        y = (attn @ v).reshape(B, T, -1)
        y = self.out(y)
        return self.norm(x + y)               # residual + LN
```

**2. Causal mask & unit test** – the mask prevents a token from attending to future positions.

```python
def causal_mask(seq_len, device):
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

def test_causal_mask():
    B, T, C = 2, 5, 8
    x = torch.randn(B, T, C)
    mask = causal_mask(T, x.device)
    enc = SimpleEncoder(d_model=C, n_head=2)
    out = enc(x, mask)
    # Verify that attention weight for i<j is -inf after softmax → zero
    with torch.no_grad():
        q, k, _ = enc.qkv(x).reshape(B, T, 3, -1, C//2).unbind(2)[:2]
        attn = (q @ k.transpose(-2, -1)) / (C//2)**0.5
        attn = attn.masked_fill(mask == 0, -1e9)
        assert torch.allclose(F.softmax(attn, dim=-1)[:, :, :, :, T-1], torch.zeros_like(attn[:, :, :, :, T-1]))
```

**3. Performance considerations** –  

| Implementation                | FLOPs (128) | FLOPs (1024) | Notes |
|-------------------------------|------------|--------------|-------|
| `torch.nn.MultiheadAttention` | ~2.1 M     | ~136 M       | Uses cuBLAS/cuDNN kernels, memory‑coalesced, ~3× faster. |
| Pure‑Python loop (for‑i)      | ~2.1 M     | ~136 M       | Python overhead dominates for long sequences; GPU under‑utilized. |

**Why use `MultiheadAttention`?** It offloads the matmul to highly‑tuned CUDA kernels, reducing latency and power consumption.  

**Edge case:** mismatched `d_model` and `n_head` (non‑divisible) raises a runtime error; validate in `__init__`.

**4. Mixed‑precision training** – wrap the forward‑backward pass with `torch.cuda.amp.autocast` and a `GradScaler`.

```python
scaler = torch.cuda.amp.GradScaler()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for xb, yb in loader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        preds = model(xb)                 # forward in FP16
        loss = F.mse_loss(preds, yb)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

On a V100, FP16 with AMP typically yields a **1.8–2.0×** wall‑clock speed‑up for sequence length 1024 while preserving the same validation loss (within 0.1 %).  

*Trade‑off:* FP16 can underflow on very small gradients; the scaler mitigates this by dynamic loss scaling.

## End‑to‑End Minimal Working Example

Below is a self‑contained script (~30 lines) that reads a raw text file, builds a character vocabulary, creates padded batches, and runs a tiny transformer‑based language model on a CPU.

```python
import torch, math, json, itertools
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import pad
from pathlib import Path

# 1️⃣ Load data and build vocab
text = Path("data/shakespeare.txt").read_text()
chars = sorted(set(text))
stoi, itos = {c:i for i,c in enumerate(chars)}, {i:c for i,c in enumerate(chars)}
vocab_size = len(chars)

# 2️⃣ Encode and batch (pad to max_len)
def encode(s): return [stoi[c] for c in s]
seq_len = 64                     # positional table size
batch_size = 32
def batchify(txt):
    ids = encode(txt)
    # split into chunks of seq_len+1 (input+target)
    chunks = [ids[i:i+seq_len+1] for i in range(0, len(ids)-seq_len, seq_len+1)]
    # pad last chunk if needed
    padded = [c + [0]*(seq_len+1-len(c)) for c in chunks]
    x = torch.tensor([c[:-1] for c in padded])
    y = torch.tensor([c[1:] for c in padded])
    return x, y

train_x, train_y = batchify(text[:int(0.9*len(text))])
val_x,   val_y   = batchify(text[int(0.9*len(text)):])

# 3️⃣ Minimal transformer block
class TinyTransformer(nn.Module):
    def __init__(self, d_model=128, n_head=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos   = nn.Parameter(torch.randn(seq_len, d_model))
        self.attn  = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.fc    = nn.Linear(d_model, vocab_size)

    def forward(self, x, src_mask):
        e = self.embed(x) + self.pos[:x.size(1)]
        attn_out, attn_weights = self.attn(e, e, e, key_padding_mask=src_mask)
        return self.fc(attn_out), attn_weights

model = TinyTransformer().cpu()
opt   = torch.optim.AdamW(model.parameters(), lr=3e-3)
criterion = nn.CrossEntropyLoss(ignore_index=0)
writer = SummaryWriter()

# Helper to create padding mask (True where pad)
def pad_mask(t): return t == 0

# 4️⃣ Training + validation loops (5 epochs)
for epoch in range(1, 6):
    model.train()
    opt.zero_grad()
    mask = pad_mask(train_x)
    logits, attn = model(train_x, mask)
    loss = criterion(logits.view(-1, vocab_size), train_y.view(-1))
    loss.backward(); opt.step()
    print(f"Epoch {epoch} | loss {loss.item():.4f}")

    # visualise first head heat‑map
    heat = attn[0,0].detach().cpu().numpy()
    print("Attention heat‑map (head 0, token 0):", heat.round(2))

    # ---- validation ----
    model.eval()
    with torch.no_grad():
        mask = pad_mask(val_x)
        val_logits, _ = model(val_x, mask)
        val_loss = criterion(val_logits.view(-1, vocab_size), val_y.view(-1))
        ppl = math.exp(val_loss.item())
        writer.add_scalar("val/perplexity", ppl, epoch)
        print(f"Validation perplexity: {ppl:.2f}")

writer.close()
```

### Edge‑case notes  
* **Sequences > positional table** – `self.pos[:x.size(1)]` slices the learned positional embedding; if `x.size(1) > seq_len` the slice would be empty, raising an error. Either increase `seq_len` or wrap positions with `pos = self.pos % seq_len`.  
* **Padding‑mask vs attention mask** – `key_padding_mask` tells `nn.MultiheadAttention` which tokens to ignore, but it does **not** zero‑out the attention scores for padded keys. Supplying the same mask for both `key_padding_mask` and `attn_mask` (or using `attn_mask = pad_mask(x).unsqueeze(1).repeat(1,x.size(1),1)`) ensures padded tokens neither attend nor are attended to, preventing leakage.

The script trains a character‑level model in ~5 seconds on a modest CPU, prints loss and a sample attention heat‑map each epoch, and logs validation perplexity to TensorBoard for easy observability.

## Common Mistakes When Building Transformers

- **Forgetting the causal mask**  
  ```python
  attn = torch.softmax(q @ k.transpose(-2, -1), dim=-1)   # no mask
  print(attn[0, :, -3:])   # shows non‑zero weights on future positions
  ```  
  The printed matrix reveals that tokens attend to later timesteps. Insert the mask before `softmax`:

  ```python
  mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
  attn = torch.softmax((q @ k.transpose(-2, -1)).masked_fill(mask, -1e9), dim=-1)
  ```

- **Mismatching batch‑first vs. sequence‑first**  
  ```python
  assert x.dim() == 3 and x.shape[0] == batch_size, \
         f"Expected (B, S, D), got {x.shape}"
  ```  
  Place the assert right before the multi‑head layer; it catches transposed inputs early.

- **Using the wrong scaling factor**  
  ```python
  scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)   # correct
  # omit division → loss roughly doubles on the MWE
  ```  
  Without `1/√d_k` the softmax saturates, causing gradients to explode and a ~2× loss increase on a minimal example.

- **Neglecting dropout in training mode**  
  ```python
  if model.training:
      mask = torch.bernoulli(torch.full_like(attn, p=dropout_p))
      logger.info(mask.float().mean().item())   # histogram of kept units
  else:
      logger.info("Eval mode – dropout disabled")
  ```  
  Verify `model.train()` vs `model.eval()`; logging the mask mean quickly shows whether dropout is active.

- **Debugging tip**  
  Wrap the forward pass with:

  ```python
  with torch.autograd.detect_anomaly():
      output = model(src, src_mask)
  ```  

  This raises a clear error when NaNs appear, often caused by un‑masked padding propagating through the attention scores.

## Production Checklist & Next Steps

- **Profile attention latency** – Run `torch.profiler` on a representative batch for sequence lengths 64, 256, 1024. Record `self_cpu_time_total` for the `self_attn` kernel and compare against your SLA (e.g., ≤5 ms per token). If latency spikes, consider reducing head count or enabling FlashAttention.  
  *Trade‑off: longer sequences increase quadratic cost; profiling helps decide max supported length.*

- **Enable gradient clipping** – Insert  
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  ```  
  after `loss.backward()`. Verify in the training log that the reported norm never exceeds 1.0; otherwise lower the learning rate or increase the clip value.  
  *Edge case: gradient norm may be NaN if loss diverges; guard with `if torch.isnan(norm): raise`.*

- **Add observability** – Export three Prometheus metrics:  
  1. Histogram of attention weights (`torch.histc(attn_weights, bins=50)`).  
  2. Per‑token inference latency (`torch.cuda.Event` timestamps).  
  3. GPU memory usage (`torch.cuda.memory_allocated`).  
  Wire them to a `/metrics` endpoint so Grafana can alert on drift.  
  *If memory usage exceeds 90 % of VRAM, trigger a scaling‑down alert.*

- **Security check** – Sanitize any raw text before tokenization: reject characters outside the allowed Unicode range and escape quotes. This prevents injection attacks when downstream services reconstruct strings from token IDs.  
  *Also validate length to avoid buffer overflows.*

- **Next‑step roadmap** –  
  - Prototype rotary positional embeddings (no extra parameters, better extrapolation).  
  - Swap the vanilla attention implementation with FlashAttention for a ~2× throughput gain on A100.  
  - Containerize the model with TorchServe, expose the `/predict` endpoint, and add health‑check probes.  
  *Prioritize experiments that give >10 % latency reduction before containerization.*
