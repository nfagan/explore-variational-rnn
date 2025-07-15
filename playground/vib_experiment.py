import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

class Bottleneck(nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.linear = nn.Linear(1, 2)

  def forward(self, x):
    mu_logvar = self.linear(x)
    mu = mu_logvar[:, 0]
    logvar = mu_logvar[:, 1]
    z = torch.randn((x.shape[0],)) * torch.exp(logvar) + mu
    return z[:, None], mu, logvar
  
class Predictor(nn.Module):
  def __init__(self, *, num_classes):
    super().__init__()
    self.linear = nn.Linear(1, num_classes)

  def forward(self, z):
    logits = self.linear(z)
    return torch.softmax(logits, dim=1)

def main():
  num_classes = 2

  encoder = Bottleneck()
  decoder = Predictor(num_classes=num_classes)

  ys = torch.arange(num_classes, dtype=torch.float32)
  vars = torch.ones_like(ys)
  mus = ys * 4.0

  xs = []
  ts = []
  for i, y in enumerate(ys):
    x = torch.randn((1000,)) * vars[i] + mus[i]
    xs.append(x)
    ts.append(y * torch.ones_like(x))
  x = torch.concatenate(xs)[:, None]
  y = torch.concatenate(ts).type(torch.long)

  optim = torch.optim.Adam([*encoder.parameters()] + [*decoder.parameters()], lr=1e-3)
  beta = 0.1

  for i in range(100000):
    z, mu, logvar = encoder(x)
    yhat = decoder(z)

    ce_loss = F.cross_entropy(yhat, y, reduction='sum')
    kl_div = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())

    loss = ce_loss + beta * kl_div
    optim.zero_grad()
    loss.backward()
    optim.step()

    if i % 100 == 0: print(f'Loss: {loss.item()}')

if __name__ == '__main__':
  main()
