# %%
import os
import time
import math
import pickle
from contextlib import nullcontext

from typing import Dict, Union, List, TypedDict, cast

import numpy as np
from torch.utils.tensorboard import SummaryWriter

# from model_new import GPTConfig, GPT
from model import GPTConfig, GPT

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# %%
##############################
# unlike nanoGPT, we will use dataclasses to control our config
# setup config
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # I/O
    out_dir: str = 'out-web'
    eval_only: bool = False
    eval_interval: int = 100 # when to run eval (saved)
    eval_iters: int = 200 # per eval no. of steps
    log_interval: int = 10 # when to print update (not saved)
    log_dir: str = 'tensorboard_log'
    always_save_on_checkpoint: bool = False
    init_from = 'resume'

    # logging
    tensorboard_log: bool = True # disable for now

    # dataset
    # these make the total batch size be ~0.5M
    # Original setting: 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
    dataset: str = 'openwebtext'
    gradient_accumulation_steps: int = 32 # to increase to 0.5M
    batch_size: int = 16 # microbatch size if grad accumulate
    block_size: int = 1024

    # baby GPT model :)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0 # pretraining set to 0
    bias: bool = False

    # adamw optimizer
    learning_rate: float = 6e-4 # with baby networks can afford to go a bit higher
    max_iters: int = 600000 # 600k * 0.5M = 300B total tokens
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip = 1.0

    # learning rate decay
    decay_lr: bool = True
    warmup_iters: int = 10 # not super necessary potentially
    lr_decay_iters: int = max_iters # make equal to max_iters usually
    min_lr: float = 6e-5 # learning_rate / 10 usually


    # system
    device: str = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = True # use PyTorch 2.0 to compile the model to be faster



config = Config()
print(config)

# %%
# init for single gpu
master_process = True
seed_offset = 0
ddp_world_size = 1

tokens_per_iter = config.gradient_accumulation_steps * ddp_world_size * config.batch_size * config.block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(config.out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in config.device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    'float32': torch.float32, 
    'bfloat16': torch.bfloat16,
    'float16': torch.float16
    }[config.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init tensorboard
writer = SummaryWriter(log_dir=config.log_dir)


# %%
##############################
# so called "poor man's" dataloader
data_dir = os.path.join('../data', config.dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(config.device, non_blocking=True), y.pin_memory().to(config.device, non_blocking=True)
    else:
        x, y = x.to(config.device), y.to(config.device)
    return x, y


# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
# meta.pkl constians the following: vocab_size, itos, stoi
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# %%
#######################################
# init the model
class GPTArgs(TypedDict):
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    bias: bool

model_args: GPTArgs = dict(
    n_layer=config.n_layer,
    n_head=config.n_head,
    n_embd=config.n_embd,
    block_size=config.block_size,
    bias=config.bias,
    vocab_size=50304, # default
    dropout=config.dropout) # start with model_args from command line

# to reduce complexity, we only assume from scratch training always
# init a new model from scratch
print("Initializing a new model from scratch")
# determine the vocab size we'll use for from-scratch training
if meta_vocab_size is None:
    print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

if config.init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif config.init_from == 'resume':
    print(f"Resuming training from {config.out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
        
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif config.init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {config.init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=config.dropout)
    model = GPT.from_pretrained(config.init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
#
# crop down the model block size if desired, using model surgery
if config.block_size < model.config.block_size:
    model.crop_block_size(config.block_size)
    model_args['block_size'] = config.block_size # so that the checkpoint will have the right value
model.to(config.device)

# force bfloat16 params for Embeddings
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()



# %%
#########################################
# runtime settings

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler('cuda', enabled=(config.dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(
    config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type)
checkpoint = None # free up memory

# compile the model
if config.compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = cast(GPT, torch.compile(model)) # , fullgraph=False)) # requires PyTorch 2.0


# helps estimate an arbitrarily accurate loss over either split using many batches
# there is no need to compensate here as model.eval() will cause self.training to be False
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss_tensor = model(X, Y)
                losses[k] = loss_tensor.mean().item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

# %%
#################################################################

# init these up here, can be overriden next time with 'resume'
iter_num = 0
best_val_loss = 1e9

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if config.decay_lr else config.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    # eval_interval = 2000
    if iter_num % config.eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if config.tensorboard_log:
            writer.add_scalars(
                'openweb',
                {
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu,
                },
                iter_num)
        # save when val_loss improves
        if losses['val'] < best_val_loss or config.always_save_on_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {config.out_dir}")
                torch.save(checkpoint, os.path.join(config.out_dir, 'ckpt.pt'))
    if iter_num == 0 and config.eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(config.gradient_accumulation_steps):
        with ctx:
            logits, loss_tensor = model(X, Y)
            # we take the sum of the loss for more pronounced gradient signals
            loss = loss_tensor.sum() / config.gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # clip the gradient
    if config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()

    # introduce a gradient analyzer to track over/underflow for fp8 kernel
    # grad = model.lm_head.weight.grad
    # if grad is not None:
    #     abs_max = grad.abs().max().item()
    #     abs_min = grad.abs().min().item()
    #     # print(f"[FP8 Grad Check] lm_head weight grad:")
    #     # print(f"  abs max: {abs_max:.3e}, abs min: {abs_min:.3e}")

    #     if abs_max < 0.002:
    #         print("  ⚠️ likely underflow")
    #     elif abs_max > 448:
    #         print("  ⚠️ likely overflow")

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % config.log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # loss is a sum of all token losses, but already factored in the number of mini-batches
        lossf = loss_tensor.mean().item()
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(config.batch_size * config.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu:.2f} tflops")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > config.max_iters:
        break
