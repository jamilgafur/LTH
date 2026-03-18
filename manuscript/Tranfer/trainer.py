import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import os
import glob

# -------------------------
# Training and Evaluation
# -------------------------

def train_one_epoch(model, train_loader, optimizer, device, scaler=None, use_autocast=False):
    """
    Performs one epoch of training.
    Supports standard FP32 training or Mixed Precision (quant) training via scaler.
    Safely handles tuple outputs from models like InceptionNet.
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    
    total_loss = 0
    correct = 0
    total = 0

    # Using tqdm for a progress bar that cleans up after itself
    pbar = tqdm.tqdm(train_loader, desc="Training", leave=False)
    
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        # Mixed Precision Path
        if use_autocast and scaler is not None:
            # Fixed deprecation warning for autocast
            with torch.amp.autocast('cuda'):
                preds = model(xb)
                
                # SMART LOSS DELEGATION
                if hasattr(model, 'compute_loss'):
                    loss = model.compute_loss(preds, yb, loss_fn)
                else:
                    loss = loss_fn(preds, yb)
            
            # Scales loss, calls backward, then unscales and steps optimizer
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

        # Metrics tracking
        # TUPLE EXTRACTION: Safely grab the main predictions if the model returns a tuple
        main_preds = preds[0] if isinstance(preds, tuple) else preds

        total_loss += loss.item() * xb.size(0)
        _, predicted = main_preds.max(1)
        correct += (predicted == yb).sum().item()
        total += yb.size(0)
        
        # Update progress bar description
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / total
    avg_acc = 100.0 * correct / total
    
    return avg_loss, avg_acc
# def train_one_epoch(model, train_loader, optimizer, device, scaler=None, use_autocast=False):
#     """
#     Performs one epoch of training.
#     Supports standard FP32 training or Mixed Precision (quant) training via scaler.
#     Safely handles tuple outputs from models like InceptionNet.
#     """
#     model.train()
#     loss_fn = nn.CrossEntropyLoss()
    
#     total_loss = 0
#     correct = 0
#     total = 0

#     # Using tqdm for a progress bar that cleans up after itself
#     pbar = tqdm.tqdm(train_loader, desc="Training", leave=False)
    
#     for xb, yb in pbar:
#         xb, yb = xb.to(device), yb.to(device)
#         optimizer.zero_grad()

#         # Mixed Precision Path
#         if use_autocast and scaler is not None:
#             # Fixed deprecation warning for autocast
#             with torch.amp.autocast('cuda'):
#                 preds = model(xb)
                
#                 # SMART LOSS DELEGATION
#                 if hasattr(model, 'compute_loss'):
#                     loss = model.compute_loss(preds, yb, loss_fn)
#                 else:
#                     loss = loss_fn(preds, yb)
            
#             # Scales loss, calls backward, then unscales and steps optimizer
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
        
#         # Standard FP32 Path
#         else:
#             preds = model(xb)
            
#             # SMART LOSS DELEGATION
#             if hasattr(model, 'compute_loss'):
#                 loss = model.compute_loss(preds, yb, loss_fn)
#             else:
#                 loss = loss_fn(preds, yb)
                
#             loss.backward()
#             optimizer.step()

#         # Metrics tracking
#         # TUPLE EXTRACTION: Safely grab the main predictions if the model returns a tuple
#         main_preds = preds[0] if isinstance(preds, tuple) else preds

#         total_loss += loss.item() * xb.size(0)
#         _, predicted = main_preds.max(1)
#         correct += (predicted == yb).sum().item()
#         total += yb.size(0)
        
#         # Update progress bar description
#         pbar.set_postfix({"loss": f"{loss.item():.4f}"})

#     avg_loss = total_loss / total
#     avg_acc = 100.0 * correct / total
    
#     return avg_loss, avg_acc
   
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
