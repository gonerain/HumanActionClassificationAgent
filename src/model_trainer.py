import os
from glob import glob
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class SkeletonDataset(Dataset):
    """Dataset wrapping skeleton sequences saved as npz files."""

    def __init__(self, files: List[str]):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx])
        seq = data["data"].astype(np.float32)
        label = int(data["label"])
        return torch.from_numpy(seq), label


def load_dataset(data_dir: str, split: float = 0.8):
    files = glob(os.path.join(data_dir, "*.npz"))
    np.random.shuffle(files)
    split_idx = int(len(files) * split)
    return files[:split_idx], files[split_idx:]


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int = 99, hidden_size: int = 128, num_classes: int = 2, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, T, N)
        _, (h, _) = self.lstm(x)
        out = self.fc(h.squeeze(0))
        return out


def train(data_dir: str, epochs: int = 20, batch_size: int = 8, lr: float = 1e-3):
    train_files, val_files = load_dataset(data_dir)
    train_loader = DataLoader(
        SkeletonDataset(train_files), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(SkeletonDataset(val_files), batch_size=batch_size)

    sample = np.load(train_files[0])["data"]
    input_size = sample.shape[-1] * sample.shape[-2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(input_size=input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

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
        print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for seq, label in val_loader:
                seq = seq.view(seq.size(0), seq.size(1), -1).to(device)
                label = label.long().to(device)
                output = model(seq)
                pred = output.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)
        acc = correct / total if total > 0 else 0
        print(f"Validation Accuracy: {acc:.4f}")

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("weights", "model.pt"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train action classifier")
    parser.add_argument("data_dir", help="directory with npz sequences")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size, args.lr)
