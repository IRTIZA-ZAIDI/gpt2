from dataclasses import dataclass

# A dataclass is a Python feature that automatically creates boilerplate code for simple data-holding classes.
# It saves you from writing __init__, __repr__, __eq__, etc. by hand.
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect

# nullcontext = “pretend there is a context manager, but don’t actually change anything.”
# nullcontext() is a context manager that does nothing on enter and exit.
# This avoids duplicating code.
from contextlib import nullcontext

# -----
# FIXING UGLY NUMBERS
# scan ur code to identify nice numbers(2^n) and ugly numbers
# increase the number of ugly numbers to match nearest power of 2 so that gpu block tiling can be optimised
# example we will override GPTConfig vocab_size to 50257 -> 50304
# dt increases from 96ms to 93ms, even though we increased vocab_size that are never going to be used.
# we introduced few useless tokens.(their probability in lm_head is forced to be 0)
# -----

# -----
# ADDING GPT-3 OPTIMIZATION PARAMTERS
# gpt3 has almost same architecture but alot more training details
# only difference is more data, longer context length, and larger model
# so we can borrow optimization/parameters mentioned in gpt3 for gpt2 training
# -----
# GPT-3 GRADUAL BATCH SIZE INCREASE
# we skip this part to keep math clean
# -----
# DATA BACH ARE USEDWITHOUT REPLACEMENT FOR EACH EPOCH
# we did this as we are moving a window over out batch


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
        # added flag to initialise weight scaling
        self.c_proj.NANOGPT_SCALE_INIT = 1
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

        # -------
        # FLASH ATTENTION
        # -------
        # Problems with naive attention:
        # - Materializes the full (T, T) attention matrix
        # - High HBM / VRAM reads & writes (memory-bound)
        # - torch.compile cannot fuse these ops because FlashAttention
        #   is not a simple kernel fusion but an algorithmic rewrite
        #
        # FlashAttention instead:
        # - Computes attention in tiled blocks (no full (T, T) matrix)
        # - Uses online softmax (numerically stable)
        # - Fuses matmul + softmax + dropout + matmul into one kernel
        # - Significantly reduces memory IO
        #
        # (B, nh) are treated as batch dimensions for attention
        # -------
        # NAIVE ATTENTION
        # -------
        # # attention (materializes the large (T, T) matrix for all the queries and keys)
        # # (B, nh) together form the batched dimensions for attention
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) = (B, nh, T, hs)
        # -------
        # FLASH ATTENTION
        # -------
        # compund operation
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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
        # added flag to initialise weight scaling
        self.c_proj.NANOGPT_SCALE_INIT = 1

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

        # init params
        # apply is a method of nn.Module
        # Call this function on every submodule in the model, recursively
        # So PyTorch internally does something like:
        # for module in self.modules():
        #       self._init_weights(module)
        #
        self.apply(self._init_weights)

    # Where does module come from?
    # is called by apply.
    # v self.apply(self._init_weights) works because PyTorch automatically traverses every submodule and passes each one as module to your function — you never pass it manually.
    # This initialization keeps the Transformer close to an identity function at the start, prevents attention saturation, stabilizes gradients through deep residual stacks, and matches the empirically proven GPT-2 setup.
    # std of xavier init is also approx same which is 1/sqrt(in_features) ~ 0.02
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # to preventing residual streams from blowing up in deep Transformer
            # Uncontrolled residuals → exploding activations → unstable training
            # Because each Transformer block has two residual paths: Attention residual and MLP residual. So total residual variance grows with ~2L.
            # (2 * n_layer)^(-0.5) scales residual-producing weights so that, despite many residual additions, the overall activation variance stays stable instead of exploding as depth increases.
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head, n_embdd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
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

    # Configure AdamW optimizer with correct weight decay handling.
    # apply decay only to true weight matrices (Linear/Embedding),
    # exclude biases and LayerNorm params for stability, and
    # automatically enable fused AdamW on CUDA for faster training.
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups.
        # any parameter that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(
            f"num decayed parameter tensors: {len(decay_params)}, "
            f"with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, "
            f"with {num_nodecay_params:,} parameters"
        )

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )

        return optimizer

# -----------------------------------------------------------------------------
import tiktoken
import numpy as np
import os
import time


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)  # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

# More precisely:
# 1. **Run model once** on each of the 4 sequences (prompt+completion) → `logits` for **every position**.
# 2. **Shift** so position `t` logits are compared to the **next token** `t+1` (standard next-token LM loss).
# 3. **Cross-entropy per token** (no averaging yet) → loss for each predicted next token.
# 4. **Apply the mask** (shifted) so **prompt losses become 0 / ignored**, and only completion-token losses remain.
# 5. **Average loss over completion tokens** for each of the 4 rows.
# 6. **Pick the minimum average loss** → that row’s completion is the model’s most likely.


def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(
        flat_shift_logits, flat_shift_tokens, reduction="none"
    )
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (
        mask[..., 1:]
    ).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288  # 2**19, ~0.5M, in number of tokens
B = 64  # micro batch size
T = 1024  # sequence length
assert (
    total_batch_size % (B * T * ddp_world_size) == 0
), "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train"
)
val_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val"
)

torch.set_float32_matmul_precision("high")

# create model
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
use_compile = (
    False  # torch.compile interferes with HellaSwag eval and Generation.
)
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = (
    19073  # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
)


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it >= max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# optimize!
optimizer = raw_model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device=device
)


# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:  # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = step == max_steps - 1

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "config": raw_model.config,
                    "step": step,
                    "val_loss": val_loss_accum.item(),
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(
                num_correct_norm, dtype=torch.long, device=device
            )
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen)  # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :]  # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()  # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0  # time difference in seconds
    tokens_processed = (
        train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    )
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(
            f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
        )
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()

# -----------------------------------------------
# LAUNCH WITH THIS COMMAND: torchrun --standalone --nproc_per_node=8 train_gpt2_ddp.py
# -----------------------------------------------

# -----------------------------------------------
### common crawl dataset: create by gpt(never released)- subset of internet
# used reddit mentioend links with atleast of 3 karma
# and extracted that link webpage
# noisy and not good

### red pajama dataset
# mixture of opensource datasets

### slim pajama dataset
# subset of red pajama dataset

### fineweb
# high quality common crawl dataset
# https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1

### fineweb-edu
# high quality common crawl dataset for education content
# https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
# 10B sample for gpt2 training
# To enhance FineWeb's quality, we developed an educational quality classifier using annotations generated by LLama3-70B-Instruct.
