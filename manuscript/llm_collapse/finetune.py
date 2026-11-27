# finetune.py -- post-collapse fine-tuning, the LLM counterpart of the
# post-compression fine-tuning in the CNN study (../Tranfer). Short LM training
# on the calibration corpus recovers what the collapse lost.
import random
from typing import Dict, List

import torch


def finetune(model, tokenizer, texts: List[str], steps: int = 50,
             lr: float = 1e-4, max_length: int = 128, seed: int = 42) -> Dict[str, float]:
    """Take `steps` AdamW steps on random corpus windows; returns a small report."""
    device = next(model.parameters()).device
    rng = random.Random(seed)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    model.train()
    losses = []
    for step in range(steps):
        text = rng.choice(texts)
        ids = tokenizer.encode(text)[:max_length]
        if len(ids) < 2:
            continue
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        loss = model(input_ids=inp, labels=inp).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    return {"steps": steps, "lr": lr,
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None}
