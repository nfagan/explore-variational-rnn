import sklearn.feature_extraction
import plotting
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import torch.optim
import numpy as np
import sklearn.feature_selection
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from typing import Callable, Tuple, Dict

# ------------------------------------------------------------------------------------------------

@dataclass
class ModelInterface:
  encoder_params: Callable[[], Dict]
  forward: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

# ------------------------------------------------------------------------------------------------

class RecurrentEncoder(nn.Module):
  def __init__(self, *, K: int, full_cov: bool):
    super().__init__()

    num_distribution_params = K + (2 ** K) - 1 if full_cov else 2 * K
    use_mlp = False
    hd = 1024

    self.mlp = nn.Sequential(nn.Linear(784, 1024), nn.ReLU()) if use_mlp else None
    # self.rnn = nn.RNN(1024 if use_mlp else 784, 1024)
    self.rnn = nn.RNNCell(1024 if use_mlp else 784, hd)
    self.fe = nn.Linear(hd, num_distribution_params)

    self.K = K #  bottleneck size
    self.full_cov = full_cov
    self.use_mlp = use_mlp
    self.hidden_dim = hd

  def ctor_params(self): return {'K': self.K, 'full_cov': self.full_cov}

  def kl(self, mus: torch.Tensor, L_or_sigma: torch.Tensor):
    if self.full_cov:
      for i in range(mus.shape[-1]):
        kl = kl_full_gaussian(mus.select(-1, i), L_or_sigma.select(-1, i))
        s = kl if i == 0 else s + kl
      return s / mus.shape[-1]
    else: raise NotImplementedError

  def forward(self, x: torch.Tensor, *, num_ticks: int, hx: torch.Tensor=None):
    all_zs = []
    all_mus = []
    all_ls = []

    if self.use_mlp: x = self.mlp(x)
    if hx is None: hx = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)

    for i in range(num_ticks):
      hx = self.rnn(x, hx)
      fe = self.fe(hx)
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
        all_zs.append(z.squeeze(-1))
        all_mus.append(mus)
        all_ls.append(L)
      else:
        raise NotImplementedError
        mus = fe[:, :self.K]
        sigmas = F.softplus(fe[:, self.K:] - 5.)
        eps = torch.randn_like(sigmas)
        z = eps * sigmas + mus
        return z, mus, sigmas
      
    zs = torch.stack(all_zs, dim=-1)
    mus = torch.stack(all_zs, dim=-1)
    L = torch.stack(all_ls, dim=-1)
    return zs, mus, L

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

def count_parameters(model: torch.nn.Module): 
  return sum(p.numel() for p in model.parameters() if p.requires_grad)
  
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

def forward_recurrent(
  enc: RecurrentEncoder, dec: Decoder, ims: torch.Tensor, ys: torch.Tensor, enc_params: Dict):
  """"""
  def losses_per_tick(enc, dec, zs, mus, sigmas, ys):
    nt = zs.shape[-1]
    kls = torch.zeros((nt,))
    ces = torch.zeros((nt,))
    for i in range(nt):
      yhats = dec(zs.select(-1, i))
      kls[i] = enc.kl(mus.select(-1, i).unsqueeze(-1), sigmas.select(-1, i).unsqueeze(-1))
      ces[i] = F.cross_entropy(yhats, ys, reduction='sum')
    return kls, ces
  
  batch_size = ims.shape[0]

  # zs, mus, sigmas = enc(ims, num_ticks=np.random.randint(1, 5), hx=None)
  zs, mus, sigmas = enc(ims, **enc_params)
  # yhats = dec(zs)
  yhats = dec(zs.select(-1, -1))

  L_kl = enc.kl(mus, sigmas)
  L_ce = F.cross_entropy(yhats, ys, reduction='sum')
  L = L_ce + beta * L_kl
  L /= batch_size

  est = torch.argmax(yhats, dim=1).reshape(ys.shape)
  acc = torch.sum(est == ys) / ys.numel()

  kls, ces = losses_per_tick(enc, dec, zs, mus, sigmas, ys)

  addtl = {}
  addtl['acc'] = acc
  addtl['err'] = 1. - acc
  addtl['zs'] = zs
  addtl['mus'] = mus
  addtl['sigmas'] = sigmas
  addtl['L_kl'] = L_kl
  addtl['L_ce'] = L_ce
  addtl['L_kl_per_tick'] = kls
  addtl['L_ce_per_tick'] = ces

  return L, addtl

def forward(enc: Encoder, dec: Decoder, ims: torch.Tensor, ys: torch.Tensor, enc_params: Dict):
  batch_size = ims.shape[0]

  zs, mus, sigmas = enc(ims, **enc_params)
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
  addtl['L_kl'] = L_kl
  addtl['L_ce'] = L_ce

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

def train(
  interface: ModelInterface, enc: Encoder, dec: Decoder, 
  num_epochs: int, mnist_train, batch_size: int):
  """"""
  optim = torch.optim.Adam([*enc.parameters()] + [*dec.parameters()], lr=1e-4, betas=[0.5, 0.999])

  use_lr_sched = True
  if use_lr_sched: lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.97)

  for e in range(num_epochs):
    for si, bi in enumerate(make_batch_indices(mnist_train, batch_size)):
      ims, ys = make_batch(mnist_train, bi)
      L, res = interface.forward(enc, dec, ims, ys, interface.encoder_params())
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

def discrete_entropy(ys, nc: int) -> float:
  fy = np.zeros((nc,))
  for i in range(nc): fy[i] = (ys == i).sum().item()
  fy = fy / fy.sum()
  hy = -(fy * np.log(fy))
  hy[fy == 0] = 0
  return hy.sum().item()

def evaluate_ticks(
  interface, enc: RecurrentEncoder, dec: Decoder, data_test, data_train, ctx: plotting.PlotContext):
  """"""
  ims, ys = make_batch(data_test)
  batch_size = ims.shape[0]
  
  num_ticks = 6
  accs = np.zeros((num_ticks,))
  errs = np.zeros_like(accs)
  ticks = np.arange(num_ticks) + 1.
  for t in range(num_ticks):
    print(f'{t+1} of {num_ticks}')
    _, res = interface.forward(enc, dec, ims, ys, {'num_ticks': t+1, 'hx': None})
    accs[t] = res['acc']
    errs[t] = res['err']
  
  # evaluate with maximum ticks
  _, res = interface.forward(enc, dec, ims, ys, {'num_ticks': num_ticks, 'hx': None})

  # summarize change in cov over ticks
  covs = np.zeros((3, num_ticks))
  areas = np.zeros((1, num_ticks))
  anisos = np.zeros_like(areas)
  mis_zy = np.zeros((res['zs'].shape[1], num_ticks))
  mis = np.zeros((2, num_ticks))

  # entropy of targets
  hy = discrete_entropy(ys, dec.nc)

  for t in range(num_ticks):
    zs = res['zs'].select(-1, t)
    mi = sklearn.feature_selection.mutual_info_classif(
      zs.detach().cpu().numpy(), ys.detach().cpu().numpy())
    mis_zy[:, t] = mi

    mis[0, t] = res['L_kl_per_tick'][t] / batch_size       # I(X; Z)
    mis[1, t] = hy - res['L_ce_per_tick'][t] / batch_size  # I(Z; Y)

    L = res['sigmas'].select(-1, t)
    cov = L @ L.transpose(-1, -2)
    covs[0, t] = cov[:, 0, 0].abs().mean()
    covs[1, t] = cov[:, -1, -1].abs().mean()
    covs[2, t] = cov[:, 1, 0].abs().mean()
    for i in range(cov.shape[0]):
      eigvals, eigvecs = np.linalg.eigh(cov.select(0, i).detach().cpu().numpy())
      width, height = 2 * np.sqrt(eigvals)
      mn, mx = min(width, height), max(width, height)
      areas[0, t] += np.pi * width * height
      anisos[0, t] += abs(1. - mx/mn)
    areas[0, t] /= cov.shape[0]
    anisos[0, t] /= cov.shape[0]

  # err over ticks
  plotting.plot_line(ticks, errs * 100., 'Ticks', 'Error (%)', context=ctx)
  # I(X; Z), I(Y; Z)
  plotting.plot_line(ticks, mis[0, :], 'Ticks', 'I(X; Z)', context=ctx)
  plotting.plot_line(ticks, mis[1, :], 'Ticks', 'I(Y; Z)', context=ctx)
  # covariance
  plotting.plot_lines(ticks, covs.T, ['z_11', 'z_22', 'z_12'], 'Ticks', 'Covariances', context=ctx)
  # areas
  plotting.plot_line(ticks, areas.T, 'Ticks', 'Area of ellipse', context=ctx)
  # anisos
  plotting.plot_line(ticks, anisos.T, 'Ticks', 'Anisotropy of ellipse', context=ctx)

  pi = torch.randperm(min(ys.shape[0], 1000))
  mus = res['mus'].index_select(0, pi)
  sigmas = res['sigmas'].index_select(0, pi)

  nt = mus.shape[-1]
  fig = plt.figure(1); fig.clf()
  axs = plotting.panels(nt)

  cmap = plt.get_cmap('hsv', dec.nc)(np.arange(dec.nc))[:, :3]
  lims = [-40, 40]

  dec_entropy = eval_classif_entropy(dec, lims[0], lims[1], 256)
  de = dec_entropy; de = (de - de.min()) / (de.max() - de.min())
  class_bounds = eval_classif(dec, lims[0], lims[1], 256, cmap)    
  # when entropy is low, 1 - de is close to 1, so the color is bright.
  # when entropy is high, 1 - de is close to 0, so the color is darker.
  lp = 0.125
  bg_im = class_bounds * (lp + (1. - lp) * (1. - de[:, :, None]))

  for i in range(nt):
    ax = axs[i]
    L = sigmas.select(-1, i)
    cov = L @ L.transpose(-1, -2)
    plotting.plot_embeddings(ax, mus.select(-1, i), cov, ys[pi], cmap)
    plotting.plot_image(ax, lims, lims, bg_im, cmap='gray')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(f'ticks={i+1}, test error={(errs[i] * 100.):.3f}%')

  fig.set_figwidth(15); fig.set_figheight(15)
  plotting.maybe_save_fig('classif', context=ctx)

def evaluate(interface: ModelInterface, enc: Encoder, dec: Decoder, mnist_test, mnist_train):
  enc.eval()
  dec.eval()

  eval_ims, eval_ys = make_batch(mnist_test)
  _, eval_res = interface.forward(enc, dec, eval_ims, eval_ys, interface.encoder_params())
  err_test = eval_res["err"] * 100.

  ims, ys = make_batch(mnist_train)
  _, train_res = interface.forward(enc, dec, ims, ys, interface.encoder_params())
  err_train = train_res["err"] * 100.
  print(f'train error: {err_train:.3f}% | test error: {err_test:.3f}%')

  if enc.full_cov:
    pi = torch.randperm(min(eval_ys.shape[0], 1000))
    mus = eval_res['mus'].index_select(0, pi)
    sigmas = eval_res['sigmas'].index_select(0, pi)

    if len(mus.shape) == 2:
      mus = mus.unsqueeze(2)
      sigmas = sigmas.unsqueeze(2)

    nt = mus.shape[-1]
    fig = plt.figure(1); fig.clf()
    axs = plotting.panels(nt)

    for i in range(nt):
      ax = axs[i]

      lims = [-20, 20]
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

      L = sigmas.select(-1, i)
      cov = L @ L.transpose(-1, -2)

      plotting.plot_embeddings(ax, mus.select(-1, i), cov, eval_ys[pi], cmap)
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
  do_save = True
  is_recurrent = True
  rand_ticks = True
  batch_size = 100
  num_epochs = 100

  root_p = os.path.join(os.getcwd(), 'data')
  cp_p = os.path.join(root_p, 
    f'checkpoint-beta_{beta:0.3f}-full_cov_{full_cov}-recurrent_{is_recurrent}-rand_ticks_{rand_ticks}.pth')

  mnist_train = datasets.mnist.MNIST(root_p, download=True, train=True, transform=ToTensor())
  mnist_test = datasets.mnist.MNIST(root_p, download=True, train=False, transform=ToTensor())

  enc_fn = RecurrentEncoder if is_recurrent else Encoder

  rec_encoder_params_fn = lambda: {
    'num_ticks': np.random.randint(1, 5) if do_train else 6, 'hx': None
  }
  interface = ModelInterface(
    forward=forward_recurrent if is_recurrent else forward,
    encoder_params=rec_encoder_params_fn if is_recurrent else lambda: {}
  )

  if do_train:
    enc = enc_fn(K=K, full_cov=full_cov)
    dec = Decoder(K=K, nc=nc)
  else:
    cp = torch.load(cp_p)
    enc = enc_fn(**cp['enc_params']); enc.load_state_dict(cp['enc_state'])
    dec = Decoder(**cp['dec_params']); dec.load_state_dict(cp['dec_state'])

  print(f'Num encoder params: {count_parameters(enc)}')
  print(f'Num decoder params: {count_parameters(dec)}')

  if do_train:
    train(interface, enc, dec, num_epochs, mnist_train, batch_size)
    if do_save:
      cp = {
        'enc_state': enc.state_dict(), 
        'dec_state': dec.state_dict(), 
        'enc_params': enc.ctor_params(), 
        'dec_params': dec.ctor_params()
      }
      torch.save(cp, cp_p)
  else:
    ctx = plotting.PlotContext()
    ctx.subdir = f'beta_{beta}'
    evaluate_ticks(interface, enc, dec, mnist_test, mnist_train, ctx)
    # evaluate(interface, enc, dec, mnist_test, mnist_train)