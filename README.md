# GPT Experiments

This repo contains my experiments with [nanogpt](https://github.com/karpathy/nanoGPT).

The `char_demo` folder contains the character-tokenization demo from nanogpt.
This uses the original implementation from nanogpt.

The `latent_attention` folder extends nanogpt by implementing Multi-head Latent
Attention (MLA) from [DeepSeek V2](https://arxiv.org/abs/2405.04434). It also
includes a decoupled RoPE that is needed to work with MLA.
