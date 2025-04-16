import plotting
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------------------------------------------

class Encoder(nn.Module):
  def __init__(self, *, K: int, full_cov: bool):
    super().__init__()

    num_distribution_params = K + (2 ** K) - 1 if full_cov else 2 * K

    self.mlp = nn.Sequential(
      nn.Linear(784, 1024),
      nn.ReLU(),
      nn.Linear(1024, 1024),
      nn.ReLU(),
      nn.Linear(1024, num_distribution_params),
      # nn.ReLU()
    )
    self.K = K #  bottleneck size
    self.full_cov = full_cov

  def ctor_params(self): return {'K': self.K, 'full_cov': self.full_cov}

  def kl(self, mus: torch.Tensor, L_or_sigma: torch.Tensor):
    if self.full_cov:
      return kl_full_gaussian(mus, L_or_sigma)
    else:
      return kl_diagonal_gaussian(mus, L_or_sigma)

  def forward(self, x):
    fe = self.mlp(x)
    mus = fe[:, :self.K]

    if self.full_cov:
      # build Cholesky factor
      sigmas = fe[:, self.K:]
      L = torch.zeros((x.shape[0], self.K, self.K))
      tril_indices = torch.tril_indices(self.K, self.K, 0)
      L[:, tril_indices[0], tril_indices[1]] = sigmas
      # softplus on diagonal
      diag_idx = torch.arange(self.K)
      L[:, diag_idx, diag_idx] = F.softplus(L[:, diag_idx, diag_idx])
      # covariance and sample
      # cov = L @ L.transpose(-1, -2)
      eps = torch.randn_like(mus).unsqueeze(-1)
      z = mus.unsqueeze(-1) + L @ eps
      return z.squeeze(-1), mus, L
    else:
      mus = fe[:, :self.K]
      sigmas = F.softplus(fe[:, self.K:] - 5.)
      eps = torch.randn_like(sigmas)
      z = eps * sigmas + mus
      return z, mus, sigmas

class Decoder(nn.Module):
  def __init__(self, *, K: int, nc: int):
    super().__init__()
    self.mlp = nn.Sequential(nn.Linear(K, nc))
    self.K = K
    self.nc = nc
  
  def forward(self, x):
    return F.softmax(self.mlp(x), dim=1)
  
  def ctor_params(self): return {'K': self.K, 'nc': self.nc}

# ------------------------------------------------------------------------------------------------
  
def kl_diagonal_gaussian(mus: torch.Tensor, sigmas: torch.Tensor):
  vars = sigmas.pow(2)
  return -0.5 * torch.sum(1.0 + vars.log() - mus.pow(2) - vars)

def kl_full_gaussian(mu: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
  k = mu.size(-1)
  # trace(Σ) = ‖L‖_F² because trace(L Lᵀ) = Σᵢⱼ L_{ij}²
  trace_term = (L ** 2).sum(dim=(-2, -1))
  # log|Σ| = 2·Σ_i log L_ii (diagonal of a Cholesky factor is > 0)
  log_det = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
  quad_term = mu.pow(2).sum(-1) # μᵀμ
  kl = 0.5 * (trace_term + quad_term - k - log_det)
  return kl.sum()

def forward(enc: Encoder, dec: Decoder, ims: torch.Tensor, ys: torch.Tensor):
  batch_size = ims.shape[0]

  zs, mus, sigmas = enc(ims)
  yhats = dec(zs)

  L_kl = enc.kl(mus, sigmas)
  L_ce = F.cross_entropy(yhats, ys, reduction='sum')
  L = L_ce + beta * L_kl
  L /= batch_size

  est = torch.argmax(yhats, dim=1).reshape(ys.shape)
  acc = torch.sum(est == ys) / ys.numel()

  addtl = {}
  addtl['acc'] = acc
  addtl['err'] = 1. - acc
  addtl['zs'] = zs
  addtl['mus'] = mus
  addtl['sigmas'] = sigmas

  return L, addtl

# ------------------------------------------------------------------------------------------------

def make_batch_indices(mnist, batch_size):
  si = torch.randperm(len(mnist))
  num_batches = len(mnist) // batch_size
  batches = []
  for i in range(num_batches):
    i0 = i * batch_size
    i1 = i0 + batch_size if i + 1 < num_batches else len(mnist)
    batches.append(si[i0:i1])
  return batches
  
def make_batch(mnist, si: torch.Tensor = None):
  if si is None: si = torch.arange(len(mnist))
  ims = []
  ys = []
  for i in si:
    im, y = mnist[i]
    ims.append(im.squeeze(0))
    ys.append(y)
  ims = torch.stack(ims, dim=0).flatten(1) * 2. - 1.  # -1 to 1
  ys =  torch.tensor(ys)
  return ims, ys

def train(enc: Encoder, dec: Decoder, num_epochs: int, mnist_train, batch_size: int):
  optim = torch.optim.Adam([*enc.parameters()] + [*dec.parameters()], lr=1e-4, betas=[0.5, 0.999])

  use_lr_sched = True
  if use_lr_sched: lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.97)

  for e in range(num_epochs):
    for si, bi in enumerate(make_batch_indices(mnist_train, batch_size)):
      ims, ys = make_batch(mnist_train, bi)
      L, res = forward(enc, dec, ims, ys)
      optim.zero_grad()
      L.backward()
      optim.step()
      acc = res["acc"] * 100.
      if si % 100 == 0: print(f'{si} | {e+1} of {num_epochs} | Loss: {L.item():.3f} | Acc: {acc:0.3f}%')
    if use_lr_sched and e % 2 == 0: lr_sched.step()

# ------------------------------------------------------------------------------------------------

def eval_classif(dec: Decoder, l0: float, l1: float, np: int, cmap: np.ndarray):
  x = torch.linspace(l0, l1, np)
  res = torch.zeros((np, np, 3))
  # n = 100
  for i in range(np):
    for j in range(np):
      z = torch.Tensor([x[j], x[i]]).unsqueeze(0)
      ez = dec(z).detach()
      res[i, j, :] = 0.5 * torch.tensor(cmap[torch.argmax(ez), :])
  return res

def eval_classif_entropy(dec: Decoder, l0: float, l1: float, np: int):
  x = torch.linspace(l0, l1, np)
  if False:
    xx, yy = torch.meshgrid(x, x)
    zs = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    yh = dec(zs).detach()
    entropy = -torch.sum(yh * yh.log(), dim=1)
    # @TODO: Double check this ordering
    res = entropy.unflatten(0, (np, np)).transpose(-1, -2)

  res2 = torch.zeros((np, np))
  for i in range(np):
    for j in range(np):
      # res2[i, j] = i/100.0 * j/100.0
      # res2[-i, -j] = (i/100.0) * 0.5
      z = torch.Tensor([x[j], x[i]]).unsqueeze(0)
      expect_z = dec(z).detach()
      e = expect_z * expect_z.log()
      e[expect_z == 0.] = 0.
      res2[i, j] = -torch.sum(e)

  r = res2
  return r

def evaluate(enc: Encoder, dec: Decoder, mnist_test, mnist_train):
  enc.eval()
  dec.eval()

  eval_ims, eval_ys = make_batch(mnist_test)
  _, eval_res = forward(enc, dec, eval_ims, eval_ys)
  err_test = eval_res["err"] * 100.

  ims, ys = make_batch(mnist_train)
  _, train_res = forward(enc, dec, ims, ys)
  err_train = train_res["err"] * 100.
  print(f'train error: {err_train:.3f}% | test error: {err_test:.3f}%')

  if enc.full_cov:
    fig = plt.figure(1)
    fig.clf()
    ax = plt.gca()

    lims = [-15, 15]
    dec_entropy = eval_classif_entropy(dec, lims[0], lims[1], 256)
    de = dec_entropy; de = (de - de.min()) / (de.max() - de.min())
    # if True: de = 1. - de

    cmap = plt.get_cmap('hsv', dec.nc)(np.arange(dec.nc))[:, :3]
    class_bounds = eval_classif(dec, lims[0], lims[1], 256, cmap)

    # bg_im = class_bounds
    # bg_im = de
    
    # when entropy is low, 1 - de is close to 1, so the color is bright.
    # when entropy is high, 1 - de is close to 0, so the color is darker.
    lp = 0.125
    bg_im = class_bounds * (lp + (1. - lp) * (1. - de[:, :, None]))

    pi = torch.randperm(min(eval_ys.shape[0], 1000))
    plotting.plot_embeddings(ax, eval_res['mus'][pi, :], eval_res['sigmas'][pi, :, :], eval_ys[pi], cmap)
    plotting.plot_image(ax, lims, lims, bg_im, cmap='gray')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(f'test error: {err_test:.3f}%')
    plt.show()

# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':
  beta = 1e-3
  K = 2
  # K = 256
  full_cov = True
  nc = 10

  do_train = False
  batch_size = 100
  num_epochs = 100

  root_p = os.path.join(os.getcwd(), 'data')
  cp_p = os.path.join(root_p, f'checkpoint-beta_{beta:0.3f}-full_cov-{full_cov}.pth')

  mnist_train = datasets.mnist.MNIST(root_p, download=True, train=True, transform=ToTensor())
  mnist_test = datasets.mnist.MNIST(root_p, download=True, train=False, transform=ToTensor())

  if do_train:
    enc = Encoder(K=K, full_cov=full_cov)
    dec = Decoder(K=K, nc=nc)
    train(enc, dec, num_epochs, mnist_train, batch_size)
    if True:
      cp = {
        'enc_state': enc.state_dict(), 
        'dec_state': dec.state_dict(), 
        'enc_params': enc.ctor_params(), 
        'dec_params': dec.ctor_params()
      }
      torch.save(cp, cp_p)
  else:
    cp = torch.load(cp_p)
    enc = Encoder(**cp['enc_params']); enc.load_state_dict(cp['enc_state'])
    dec = Decoder(**cp['dec_params']); dec.load_state_dict(cp['dec_state'])
    evaluate(enc, dec, mnist_test, mnist_train)