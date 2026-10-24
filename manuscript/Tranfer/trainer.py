import sys

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import os
import glob

# -------------------------
# Training and Evaluation
# -------------------------

def train_one_epoch(model, train_loader, optimizer, device, scaler=None, use_autocast=False,
                    epoch=None, total_epochs=None):
    """
    Performs one epoch of training.
    Supports standard FP32 training or Mixed Precision (quant) training via scaler.
    Safely handles tuple outputs from models like InceptionNet.

    Progress bar is shown on interactive terminals only (auto-disabled in
    non-TTY contexts such as PBS batch logs). Metrics are accumulated on-device
    and synced at throttle points (postfix refresh) and once at epoch end.
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    desc = f"Epoch {epoch + 1}/{total_epochs}" if epoch is not None and total_epochs else "Training"
    pbar = tqdm.tqdm(
        train_loader,
        desc=desc,
        leave=False,
        disable=not sys.stderr.isatty(),
    )

    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    total = 0

    postfix_every = 20

    for step, (xb, yb) in enumerate(pbar):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Mixed Precision Path
        if use_autocast and scaler is not None:
            with torch.amp.autocast('cuda'):
                preds = model(xb)

                # SMART LOSS DELEGATION
                if hasattr(model, 'compute_loss'):
                    loss = model.compute_loss(preds, yb, loss_fn)
                else:
                    loss = loss_fn(preds, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Standard FP32 Path
        else:
            preds = model(xb)

            # SMART LOSS DELEGATION
            if hasattr(model, 'compute_loss'):
                loss = model.compute_loss(preds, yb, loss_fn)
            else:
                loss = loss_fn(preds, yb)

            loss.backward()
            optimizer.step()

        # Metrics tracking (device-side accumulation; no per-batch host sync)
        # TUPLE EXTRACTION: Safely grab the main predictions if the model returns a tuple
        main_preds = preds[0] if isinstance(preds, tuple) else preds

        total_loss += loss.detach() * xb.size(0)
        _, predicted = main_preds.max(1)
        correct += (predicted == yb).sum()
        total += yb.size(0)

        # Throttled postfix refresh (bounded syncs while the bar is live)
        if not pbar.disable and (step + 1) % postfix_every == 0:
            pbar.set_postfix({
                "loss": f"{total_loss.item() / total:.4f}",
                "acc": f"{100.0 * correct.item() / total:.2f}%",
            })

    if not pbar.disable:
        pbar.set_postfix({
            "loss": f"{total_loss.item() / total:.4f}",
            "acc": f"{100.0 * correct.item() / total:.2f}%",
        })

    avg_loss = total_loss.item() / total
    avg_acc = 100.0 * correct.item() / total

    return avg_loss, avg_acc


def evaluate(model, loader, device, quant=False): #
    model.eval() #
    correct = total = 0 #
    use_autocast = quant and device.type == 'cuda' #

    with torch.no_grad(): #
        for xb, yb in loader: #
            xb, yb = xb.to(device), yb.to(device) #
            if use_autocast: #
                with torch.cuda.amp.autocast(): #
                    preds = model(xb) #
            else:
                preds = model(xb) #

            # Even in eval mode, it's good practice to ensure we have the main predictions
            main_preds = preds[0] if isinstance(preds, tuple) else preds

            _, predicted = main_preds.max(1) #
            correct += (predicted == yb).sum().item() #
            total += yb.size(0) #

    return 100 * correct / total #
