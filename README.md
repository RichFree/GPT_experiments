# GPT Experiments

This repo contains my experiments with [nanogpt](https://github.com/karpathy/nanoGPT).

The `char_demo` folder contains the character-tokenization demo from nanogpt.
This uses the original implementation from nanogpt.

The `quantize_head` folder contains the fp8 lm_head from  the
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) implementation.
It also replaces the layer-norm of GPT2 with RMSNorm from the same repo. On the
4080 super, it achieves ~100 tflops after torch.compile. The original nanogpt
only achieves ~70 tflops after torch.compile.

Note:
- for torch.compile must be disabled for some sections in linear_fp8
