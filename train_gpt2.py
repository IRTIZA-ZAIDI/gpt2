from dataclass import dataclass

# A dataclass is a Python feature that automatically creates boilerplate code for simple data-holding classes.
# It saves you from writing __init__, __repr__, __eq__, etc. by hand.
import torch
import torch.nn as nn
from torch.nn import functional as f

# -----


# ----- GPTConfig
# this contains configs to setup our gpt-2
@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


# ----- CausalSelfAttention
# this is the CausalSelfAttention implementation for gpt-2
# this is multiheaded attention
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query and value projection for all heads, but in a batch
        # 3 means key, query and value projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not a bias but a mask / non-learnable parameter we dont need to track
        # It creates and stores a causal attention mask (lower-triangular matrix) inside a PyTorch module so the Transformer cannot attend to future tokens.
        # In multi-head self-attention, the attention score tensor is: att.shape = (B, H, T, T)
        # B = batch size, H = number of heads, T = sequence length
        # Each head in each batch has its own T×T attention matrix.
        # torch.tril(torch.ones(config.block_size, config.block_size) creates (T, T)
        # We want to apply the same causal rule to: every batch element and every attention head
        # want this mask to behave like: (B, H, T, T)
        # Broadcasting rule: dimensions are equal or one of them is 1
        # .view(1, 1, T, T) --> converts (T, T) to (1, 1, T, T)
        # .view(1, 1, T, T) makes the causal mask broadcastable so one fixed mask can be automatically applied to all batches and all attention heads without copying data.
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size() # batch_size, sequence_length and embedding dimention(n_embd)
        # calculate query, key and value for all heads in a batch and move the head forward to be batched
        # nh is 'number of heads', hs is 'head size', and C is 'number of channel' = nh * ns
        # In GPT-2 (124M), nhead=12, hs=64, C=nh*hs=768 channels in transformer
        qkv = self.c_attn(x) 
        # qkv.shape = (B, T, 3 * C)
        # split(size, dim) splits a tensor into chunks of size along dimension dim
        # Splitting along dim=2 into chunks of C gives 3 tensors:
        # q.shape = (B, T, C), k.shape = (B, T, C), v.shape = (B, T, C)
        # Each token has one big query, key, value vector
        # Heads are not separated yet
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Instead of one C-dim vector
        # We want n_head vectors, each of size head_size = C / n_head
        # So we need to split the embedding dimension into heads
        # k = k.view(B, T, self.n_head, C // self.n_head): (B, T, C) -> (B, T, n_head, head_size)
        # k = k.transpose(1, 2) -> Swaps T and n_head: (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # attention (materializes the large (T, T) matrix for all the queries and keys)
        # (B, nh) together form the batched dimensions for attention
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = f.softmax(att, dim=-1) 
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) = (B, nh, T, hs)

        # y = (B, nh, T, hs)
        # y.transpose(1,2) -> (B, nh, T, hs) -> (B, T, nh, hs)
        # .contiguous() -> After transpose, the tensor: has a non-contiguous memory layout .view() cannot be applied safely
        # .contiguous(): creates a contiguous copy in memory makes .view() legal
        # .view(B, T, C): Merge heads back into embedding dimension
        # C = nh * hs
        # (B, T, nh, hs) -> (B, T, C)
        # y.shape = (B, T, C)
        # Before attention: token → [768]
        # token → [64 | 64 | ... | 64]  (12 heads)
        #        └────── concatenated ──────┘
        y = y.transpose(1, 2).contiguous().view(B, T, C) # reassemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


# ----- MLP
# this is the mlp implementation
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        # helps to recover dead neuron due as it always contribute a gradient
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# ----- Block
# this contains Block of attention and mlp
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        # attention is a communication channel where tokens communicate with each other
        # its a aggregation/pooling function / reduced operation
        self.attn = CausalSelfAttention(conig)
        self.ln2 = nn.LayerNorm(config.n_embd)
        # mlp is just a map function
        self.mlp = MLP(config)

    def forward(self, x):
        # residual
        # addition is just branch in gradients
        # gradients flow directly through the residual path and also through the blocks
        # pre-norm version
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ----- GPT
# final class for GPT
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
