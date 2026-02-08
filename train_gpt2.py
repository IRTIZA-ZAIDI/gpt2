from dataclasses import dataclass

# A dataclass is a Python feature that automatically creates boilerplate code for simple data-holding classes.
# It saves you from writing __init__, __repr__, __eq__, etc. by hand.
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# -----


# ----- GPTConfig
# this contains configs to setup our gpt-2
@dataclass
class GPTConfig:
    block_size: int = 1024  # max seq length
    vocab_size: int = (
        50257  # number of tokens: 50,000 merges + 256 bytes + 1 <|endoftext|> token
    )
    n_layer: int = 12  # number of layer
    n_head: int = 12  # number of head
    n_embd: int = 768  # embedding dimension
    # # for mock testing
    # block_size: int = 256
    # vocab_size: int = 65
    # n_layer: int = 6
    # n_head: int = 6
    # n_embd: int = 384


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
        B, T, C = (
            x.size()
        )  # batch_size, sequence_length and embedding dimention(n_embd)
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
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # attention (materializes the large (T, T) matrix for all the queries and keys)
        # (B, nh) together form the batched dimensions for attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
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
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # reassemble all head outputs side by side

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
        self.attn = CausalSelfAttention(config)
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
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # at initialization we want all prob of tokens to be equal and not favour any token
        # we want roughly same prob for all which is approx 1/50257
        # quick sanity check of cross entropy at starting is that
        # -ln(1/50257) = 10.82..
        # this is the first loss we expect
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        # Same matrix, used in opposite directions
        # makes both layers point to the same Parameter object.
        # Same tensor, just transposed by nn.Linear internally.
        # Shapes still work because embeddings use the matrix as (vocab, emb) via row lookup, while the output head uses the transpose (emb, vocab) in matrix multiplication — the same tensor, two valid orientations.
        self.transformer.wte.weight = self.lm_head.weight 

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block_size is {self.config.block_size}"
        # forward the token and positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # postion embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb

        # forward block of transformer
        for block in self.transformer.h:
            x = block(x)

        # forward the final layernorm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # logits.shape = (B, T, V), targets.shape = (B, T)
            # Each token has: a probability distribution over V words and one target token id
            # torch.nn.functional.cross_entropy expects:
            # input  : (number of independent predictions, number of classes), target : (number of independent predictions)
            # It does not understand sequences or batches — only flat classification problems.
            # Flatten logits logits.view(-1, logits.size(-1)): (B, T, V) -> (B*T, V)
            # each row = one token prediction, total predictions = B × T
            # Flatten targets targets.view(-1): (B, T) -> (B*T)
            # one correct class id per token
            # loss is averaged over all tokens, not per sequence, not per batch — per token
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # cross_entropy expects (N, C) vs (N), so GPT flattens (B, T, V) → (B·T, V) and (B, T) → (B·T) to compute loss over all tokens at once.
        return logits, loss

    # The method is bound to the class, not to an instance
    # The first argument is the class itself, conventionally named cls
    # You can call it without creating an object first
    # self = an already-created object, You can only call this after instantiating the class
    # This pattern is called a factory method.
    # @classmethod is used when a method needs access to the class itself (not an instance), typically to create and return a new object, which is why it takes cls instead of self.
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from hugging face"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % modeltype)

        # n_layer, n_head, n_embdd are determined from model_type
        config_args = {
            "gpt2": dict(n_layers=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layers=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layers=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layers=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for gpt2 models
        config_args["block_size"] = 1024  # always 1024 for gpt2 models

        # create a from scratch initialized minGPT model
        # **config_args unpacks a dictionary into named arguments (key=value), while * unpacks a list/tuple into positional arguments.
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer

        # initialized huggingface/transformer model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match the name and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore mask/bias
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # ignore mask/bias
        # All of these were implemented as Conv1D in GPT-2.
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically openai checkpoints uses a "Conv1D" module, but we want to use vanilla
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatch keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        # GPT-2’s original Conv1D layers are actually linear layers with transposed weight storage, so when importing them into PyTorch nn.Linear, we must transpose those weights and verify the reversed shape matches exactly.
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                # What [::-1] does: It reverses the shape tuple.
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


# -----
# auto-detect device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# -----
import tiktoken


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading next batch will be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


# -----
train_loader = DataLoaderLite(4, 32)

# get logits
model = GPT(GPTConfig())
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# optimize
for i in range(2):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")

import sys

sys.exit(0)


# -----
# replicating huggingface pipeline
# num_return_sequences = 5
# max_length = 30

# model = GPT.from_pretrained("gpt2")
# print('Yay! no crash')
# model.eval()
# model.to(device)

# # prefix tokens
# import tiktoken
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a langauge model")
# tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
# tokens = token.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
# x = tokens.to(device)

# # generate! rightnow x is (B,T) where B = 5, T = 8
# # set seed to 42
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#     # forward the model to get logits
#     with torch.no_grad():
#         logits = model(x) # (B, T, vocab_size)
#         # take the logits at the last position
#         logits = logits[:,-1,:] # (B, vocab_size)
#         # get the probabilities
#         probs = f.softmax(logits, dim=-1) # (B, vocab_size)
#         # do top-k sampling of 50 (huggingface pipeline default)
#         # topk_probs here become (5,50), topk_indices is (5,50)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#         # select a token from top-k probabilities
#         # torch.multinomial(input, num_samples)
#         ix = torch.multinomial(topk_probs, 1) # (B,1)
#         # gather the corresponding indices
#         # torch.gather(input, dim, index)
#         xcol = torch.gather(topk_indices, -1, ix) # (B,1)
#         # append to sequence
#         x = torch.cat((x, xcol), dim=1)

# # print generated text
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)
