import os
from glob import glob
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report


class SkeletonDatasetV2(Dataset):
    """Dataset with velocity features."""

    def __init__(self, files: List[str]):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx])
        seq = data["data"].astype(np.float32)  # (T, J, D)
        vel = np.diff(seq, axis=0, prepend=seq[0:1])
        feat = np.concatenate([seq, vel], axis=-1)
        label = int(data["label"])
        return torch.from_numpy(feat), label


def load_dataset(data_dir: str, split: float = 0.8):
    files = glob(os.path.join(data_dir, "**", "*.npz"), recursive=True)
    np.random.shuffle(files)
    split_idx = int(len(files) * split)
    return files[:split_idx], files[split_idx:]


class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, outputs: torch.Tensor):
        # outputs: (B, T, H*2)
        weights = torch.softmax(self.attn(outputs).squeeze(-1), dim=1)
        context = torch.sum(outputs * weights.unsqueeze(-1), dim=1)
        return context


class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 1, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        context = self.attn(outputs)
        out = self.fc(context)
        return out


def train(
    data_dir: str,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    patience: int = 5,
):
    train_files, val_files = load_dataset(data_dir)
    train_loader = DataLoader(
        SkeletonDatasetV2(train_files), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(SkeletonDatasetV2(val_files), batch_size=batch_size)

    sample = np.load(train_files[0])["data"]
    vel = np.diff(sample, axis=0, prepend=sample[0:1])
    input_size = sample.shape[-1] * 2 * sample.shape[-2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMAttentionClassifier(input_size=input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for seq, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            seq = seq.view(seq.size(0), seq.size(1), -1).to(device)
            label = label.long().to(device)
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for seq, label in val_loader:
                seq = seq.view(seq.size(0), seq.size(1), -1).to(device)
                label = label.long().to(device)
                output = model(seq)
                loss = criterion(output, label)
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)
        val_loss /= len(val_loader)
        acc = correct / total if total > 0 else 0
        print(
            f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f} | Validation Loss: {val_loss:.4f} | Acc: {acc:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

    os.makedirs("weights", exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join("weights", "model_v2.pt"))

    # confusion matrix and classification report on validation set
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for seq, label in val_loader:
            seq = seq.view(seq.size(0), seq.size(1), -1).to(device)
            label = label.long().to(device)
            output = model(seq)
            pred = output.argmax(dim=1)
            y_true.extend(label.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train action classifier V2")
    parser.add_argument("data_dir", help="directory with npz sequences")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.patience)
