"""Interface for loading language models with CPU/GPU awareness."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def load_llm(
    model_name: str = "meta-llama/Llama-2-7b-chat-hf", device: int | str | None = None
):
    """Load a Hugging Face model as a text-generation pipeline."""
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    return generator
