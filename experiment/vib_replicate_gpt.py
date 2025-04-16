import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os


# ----------------------------
#  Encoder / Decoder modules
# ----------------------------
class Encoder(nn.Module):
    """MLP 784 → 1024 → 1024 → 2K that predicts mean & log‑variance."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(28 * 28, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU()
        )
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

    def forward(self, x: torch.Tensor):  # x: [B, 784]
        h = self.hidden(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Classifier(nn.Module):
    """Simple logistic‑regression decoder: K → 10 logits."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 10)

    def forward(self, z: torch.Tensor):
        return self.fc(z)


# ----------------------------
#  VIB wrapper
# ----------------------------
class VIB(nn.Module):
    def __init__(self, latent_dim: int = 256, beta: float = 1e-3):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.classifier = Classifier(latent_dim)
        self.beta = beta

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor):
        # KL[N(mu, sigma) || N(0, I)] analytic form, averaged over batch
        return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1).mean()

    def forward(self, x: torch.Tensor, y = None):
        mu, logvar = self.encoder(x)
        z = self._reparameterize(mu, logvar)
        logits = self.classifier(z)

        if y is None:
            return logits  # inference mode

        ce = nn.functional.cross_entropy(logits, y, reduction="mean")
        kl = self._kl_divergence(mu, logvar)
        loss = ce + self.beta * kl
        return loss, ce.detach(), kl.detach()


# ----------------------------
#  Training / evaluation utils
# ----------------------------
@torch.no_grad()
def evaluate(model: VIB, loader: DataLoader, device: torch.device, mc_samples: int = 12):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device).view(x.size(0), -1), y.to(device)
        # Monte‑Carlo average of logits
        logits_accum = 0.0
        for _ in range(mc_samples):
            logits_accum += model(x)
        logits = logits_accum / mc_samples
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


def train_epoch(model: VIB, loader: DataLoader, optim: torch.optim.Optimizer, device: torch.device):
    model.train()
    total_loss = total_ce = total_kl = 0.0
    for x, y in loader:
        x, y = x.to(device).view(x.size(0), -1), y.to(device)
        optim.zero_grad()
        loss, ce, kl = model(x, y)
        loss.backward()
        optim.step()
        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_ce += ce.item() * batch_size
        total_kl += kl.item() * batch_size
    n = len(loader.dataset)
    return total_loss / n, total_ce / n, total_kl / n


# ----------------------------
#  Main script
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Deep Variational Information Bottleneck on MNIST")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--latent", type=int, default=256)
    parser.add_argument("--data", type=Path, default=Path("./data"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_p = os.path.join(os.getcwd(), 'data')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_ds = datasets.MNIST(root_p, train=True, transform=transform, download=True)
    test_ds = datasets.MNIST(root_p, train=False, transform=transform, download=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

    model = VIB(latent_dim=args.latent, beta=args.beta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        loss, ce, kl = train_epoch(model, train_loader, optimizer, device)
        acc = evaluate(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
        print(
            f"Epoch {epoch:3d} | loss {loss:.4f} | ce {ce:.4f} | kl {kl:.4f} | test‑acc {acc*100:5.2f}% (best {best_acc*100:5.2f}%)"
        )

    print("Training complete. Best accuracy:", best_acc)


if __name__ == "__main__":
    main()
