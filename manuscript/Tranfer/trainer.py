import torch
import torch.nn as nn
import torch.optim as optim
# -------------------------
# Training and Evaluation
# -------------------------
def train_and_evaluate(model, train_loader, test_loader, device, epochs=10, post_compress_epochs=False):
    if epochs <= 0:
        print("[Warning] Number of training epochs is zero or negative!")
        final_acc = evaluate(model, test_loader, device)
        return {
            "accuracies": [],
            "final_accuracy": final_acc,
            "losses": [],
            "total_epochs_trained": 0,
        }

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    accuracies, losses = [], []
    total_epochs_trained = 0

    max_epochs = epochs
    patience = 5
    threshold = 0.05  # 0.05% improvement threshold
    epochs_no_improve = 0
    best_acc = 0

    epoch = 0
    while True:
        # Train one epoch
        model.train()
        total_loss = correct = total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            _, predicted = preds.max(1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)

        avg_loss = total_loss / total
        acc = 100 * correct / total

        phase = "Post-compress" if (post_compress_epochs and epoch >= epochs) else "Epoch"
        print(f"{phase} {epoch + 1}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")

        accuracies.append(acc)
        losses.append(avg_loss)
        total_epochs_trained += 1
        epoch += 1

        if post_compress_epochs and epoch > epochs:
            if acc - best_acc > threshold:
                best_acc = acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience or epoch >= epochs + 100:
                print(f"Stopping early after {epoch - epochs} post-compression epochs due to no significant improvement.")
                break
        else:
            if acc > best_acc:
                best_acc = acc

            if epoch >= epochs and not post_compress_epochs:
                break

    final_acc = evaluate(model, test_loader, device)
    print(f"Final Test Accuracy: {final_acc:.2f}%")

    return {
        "accuracies": accuracies,
        "final_accuracy": final_acc,
        "losses": losses,
        "total_epochs_trained": total_epochs_trained,
    }

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            _, predicted = preds.max(1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)
    return 100 * correct / total
