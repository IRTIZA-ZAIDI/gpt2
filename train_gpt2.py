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
        # key, query and value projection 




# ----- MLP
# this is the mlp implementation
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        # helps to recover dead neuron due as it always contribute a gradient
        self.gelu = nn.GELU(approximate='tanh')
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
