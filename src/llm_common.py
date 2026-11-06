"""
Shared pieces for E1/E2 (small language models). Models and the corpus live under followup/models and
followup/data (not tracked); results/e1_downloads.json records the exact Hugging Face revisions.

Paper one pins pillow==9.0.1 (other versions rasterise differently), which predates PIL.Image.Resampling.
transformers imports that name at load time even for text models, so it is aliased here instead of
upgrading Pillow in the shared environment. No E1/E2 code touches images.
"""
import os
import PIL.Image
if not hasattr(PIL.Image, "Resampling"):
    PIL.Image.Resampling = PIL.Image                      # old Pillow keeps NEAREST, BILINEAR, ... on the module

import common as C
import numpy as np

MODELS = os.path.join(C.FOLLOWUP, "models"); DATA = os.path.join(C.FOLLOWUP, "data")
PILE = os.path.join(DATA, "pile-10k", "data", "train-00000-of-00001-4746b8785c874cc7.parquet")

def load(name, dtype=None, device="cpu"):
    """name: 'pythia-160m' (GPT-NeoX, GELU MLP) or 'smollm2-135m' (Llama, SwiGLU MLP). Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d = os.path.join(MODELS, name); tok = AutoTokenizer.from_pretrained(d)
    m = AutoModelForCausalLM.from_pretrained(d, dtype=dtype or torch.float32).to(device).eval()
    return m, tok

def pile_texts():
    import pyarrow.parquet as pq
    return pq.read_table(PILE).column("text").to_pylist()
