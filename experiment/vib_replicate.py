import plotting
from logictask import LogicDataset as AltLogicDataset
from tasks import generate_logic_task_sequence, generate_parity_task_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import torch.optim
from torch.multiprocessing import Process
import numpy as np
import sklearn.feature_selection
import matplotlib.pyplot as plt
from scipy.io import savemat
import os
from dataclasses import dataclass
import itertools
from typing import Callable, Tuple, Dict, List

_NO_RANDOM_SEED = -1

# ------------------------------------------------------------------------------------------------

class ParityDataset(Dataset):
  def __init__(self, batch_size: int, input_dim: int):
    super().__init__()
    self.batch_size = batch_size
    self.input_dim = input_dim
  
  def __len__(self):
    return self.batch_size
  
  def __getitem__(self, idx):
    input_dim = self.input_dim
    x, y, mask = generate_parity_task_sequence(input_dim)
    y = y.type(torch.long)
    # make x: (Nx1), y: (1,)
    x = x.unsqueeze(-1)
    return x, y

class LogicDataset(Dataset):
  def __init__(self, batch_size: int, seq_len: int, num_ops: int, num_samples: int = None):
    super().__init__()
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.num_ops = num_ops
    self.samples = []
    self.num_samples = num_samples

    if num_samples is not None:
      for _ in range(num_samples):
        x, y = generate_logic_task_sequence(self.seq_len, self.num_ops)
        self.samples.append((x, y))
  
  def __len__(self):
    if self.num_samples is not None:
      return self.num_samples
    else:
      return self.batch_size
  
  def __getitem__(self, idx):
    if self.num_samples is not None:
      x, y = self.samples[idx]
    else:
      x, y = generate_logic_task_sequence(self.seq_len, self.num_ops)
    # x: (seq_len x N) | y: (seq_len x 1) | m: (seq_len x 1)
    y = y.type(torch.long)
    x = x.T
    y = y.T.squeeze(0)
    return x, y

class WrappedMNISTDataset(Dataset):
  def __init__(self, ims: torch.Tensor, ys: torch.Tensor):
    super().__init__()
    self.ims = ims
    self.ys = ys
  
  def __len__(self):
    return self.ys.shape[0]
  
  def __getitem__(self, idx):
    ims = self.ims.select(0, idx)
    ys = self.ys[idx]
    return ims, ys

# ------------------------------------------------------------------------------------------------

class EncoderParams(object):
  def get(self): return {}

class RecurrentEncoderParams(EncoderParams):
  def __init__(self, max_num_ticks: int):
    super().__init__()
    self.max_num_ticks = max_num_ticks
  def get(self): 
    return {'num_ticks': 1 + np.random.randint(self.max_num_ticks)}

@dataclass
class ModelInterface:
  encoder_params: EncoderParams
  forward: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
  random_seed: int

# ------------------------------------------------------------------------------------------------

class RecurrentEncoder(nn.Module):
  def __init__(
    self, *, input_dim: int, hidden_dim: int, K: int, full_cov: bool, beta: float, 
    rnn_cell_type: str):
    """"""
    assert rnn_cell_type in ['rnn', 'lstm']

    super().__init__()

    num_distribution_params = K + (2 ** K) - 1 if full_cov else 2 * K
    use_mlp = False
    hd = hidden_dim

    self.mlp = nn.Sequential(nn.Linear(input_dim, 1024), nn.ReLU()) if use_mlp else None
    # rnn
    rnn_input_size = 1024 if use_mlp else input_dim
    if rnn_cell_type == 'rnn':
      self.rnn = nn.RNNCell(rnn_input_size, hd)
    elif rnn_cell_type == 'lstm':
      self.rnn = nn.LSTMCell(rnn_input_size, hd)
    else:
      assert False
    
    self.fe = nn.Linear(hd, num_distribution_params)

    self.K = K #  bottleneck size
    self.full_cov = full_cov
    self.use_mlp = use_mlp
    self.hidden_dim = hd
    self.beta = beta
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.rnn_cell_type = rnn_cell_type

  def ctor_params(self): return {
    'K': self.K, 'full_cov': self.full_cov, 'input_dim': self.input_dim, 'beta': self.beta,
    'hidden_dim': self.hidden_dim, 'rnn_cell_type': self.rnn_cell_type
  }

  def sample(self, mus: torch.Tensor, L_or_sigma: torch.Tensor):
    if self.full_cov:
      raise NotImplementedError
      eps = torch.randn_like(mus).unsqueeze(-1)
      z = mus.unsqueeze(-1) + L_or_sigma @ eps
    else:
      eps = torch.randn_like(L_or_sigma)
      z = eps * L_or_sigma + mus
    return z

  def kl(self, mus: torch.Tensor, L_or_sigma: torch.Tensor):
    for i in range(mus.shape[-1]):
      mu, sig = mus.select(-1, i), L_or_sigma.select(-1, i)
      kl = kl_full_gaussian(mu, sig) if self.full_cov else kl_diagonal_gaussian(mu, sig)
      s = kl if i == 0 else s + kl
    return s / mus.shape[-1]

  def forward(self, x: torch.Tensor, *, num_ticks: int, hx: torch.Tensor=None):
    all_zs = []
    all_mus = []
    all_ls = []

    if self.use_mlp: x = self.mlp(x)
    # if hx is None: hx = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)

    for i in range(num_ticks):
      hx = self.rnn(x, hx)
      proj_hx = hx if self.rnn_cell_type == 'rnn' else hx[0]
      fe = self.fe(proj_hx)
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
        mus = fe[:, :self.K]
        sigmas = F.softplus(fe[:, self.K:] - 5.)
        eps = torch.randn_like(sigmas)
        z = eps * sigmas + mus
        all_zs.append(z)
        all_mus.append(mus)
        all_ls.append(sigmas)
      
    zs = torch.stack(all_zs, dim=-1)
    mus = torch.stack(all_mus, dim=-1)
    L = torch.stack(all_ls, dim=-1)
    return zs, mus, L, hx

# ------------------------------------------------------------------------------------------------

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
  enc: RecurrentEncoder, dec: Decoder, xs: torch.Tensor, ys: torch.Tensor, enc_params: Dict, 
  num_z_eval=1):
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
  
  if len(xs.shape) == 2:
    # make into a sequence
    xs = xs.unsqueeze(-1)
    ys = ys.unsqueeze(-1)
  
  batch_size = xs.shape[0]
  seq_len = xs.shape[-1]
  hx = None
  seq_yhats = []

  for t in range(seq_len):
    x = xs.select(-1, t)
    y = ys.select(-1, t)

    if num_z_eval == 1:
      zs, mus, sigmas, hx = enc(x, **enc_params, hx=hx)
    else:
      zs = []
      for _ in range(num_z_eval):
        zs_, mus, sigmas, hx_ = enc(x, **enc_params, hx=hx)
        zs.append(zs_)
      hx = hx_
      zs = torch.mean(torch.stack(zs, dim=-1), dim=-1)

    yhats = dec(zs.select(-1, -1))
    est = torch.argmax(yhats, dim=1).reshape(y.shape)
    acc = torch.sum(est == y) / y.numel()

    L_kl = enc.kl(mus, sigmas)
    L_ce = F.cross_entropy(yhats, y, reduction='sum')
    L = L_ce + enc.beta * L_kl
    L /= batch_size

    kls, ces = losses_per_tick(enc, dec, zs, mus, sigmas, y)

    if t == 0:
      seq_L, seq_L_kl, seq_L_ce = L, L_kl, L_ce
    else:
      seq_L, seq_L_kl, seq_L_ce = seq_L+L, seq_L_kl+L_kl, seq_L_ce+L_ce

    seq_yhats.append(yhats)

  addtl = {}
  addtl['acc'] = acc
  addtl['err'] = 1. - acc
  addtl['zs'] = zs
  addtl['mus'] = mus
  addtl['sigmas'] = sigmas
  addtl['L_kl'] = seq_L_kl
  addtl['L_ce'] = seq_L_ce
  addtl['L_kl_per_tick'] = kls
  addtl['L_ce_per_tick'] = ces
  addtl['yhat'] = torch.stack(seq_yhats, dim=-1)

  return seq_L, addtl

# ------------------------------------------------------------------------------------------------

def make_mnist_dataloader(mnist, batch_size: int):
  ims, ys = make_batch(mnist)
  d = WrappedMNISTDataset(ims, ys)
  return DataLoader(d, batch_size=batch_size)

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

def eval_train(
  interface: ModelInterface, enc: RecurrentEncoder, dec: Decoder, data_test: DataLoader, enc_p: dict):
  """"""
  enc.eval()
  dec.eval()
  exys = [(x, y) for x, y in data_test]
  ex = torch.cat([x[0] for x in exys], dim=0)
  ey = torch.cat([x[1] for x in exys], dim=0)
  L_eval, res_eval = interface.forward(enc, dec, ex, ey, enc_p)
  eval_acc = res_eval["acc"] * 100.
  enc.train()
  dec.train()
  return L_eval, eval_acc

def train(
  interface: ModelInterface, enc: RecurrentEncoder, dec: Decoder, 
  num_epochs: int, data: DataLoader, data_test: DataLoader, 
  cp_p: str, cp_name: str, do_save: bool, use_lr_sched: bool, lr: float):
  """"""
  optim = torch.optim.Adam([*enc.parameters()] + [*dec.parameters()], lr=lr, betas=[0.5, 0.999])

  if use_lr_sched: lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.97)
  if interface.random_seed != _NO_RANDOM_SEED: torch.manual_seed(interface.random_seed)

  for e in range(num_epochs):
    si = 0
    for xs, ys in data:
      enc_p = interface.encoder_params.get()
      L, res = interface.forward(enc, dec, xs, ys, enc_p)
      optim.zero_grad()
      L.backward()
      optim.step()
      acc = res["acc"] * 100.
      if si % 100 == 0:
        L_eval, eval_acc = eval_train(interface, enc, dec, data_test, enc_p)
        print(f'\t\t{e+1} of {num_epochs} | Loss: {L.item():.3f} | Acc: {acc:0.3f}% | ' + \
              f'Eval loss: {L_eval.item():0.3f} | Eval acc: {eval_acc:0.3f}%', flush=True)
      si += 1
    if use_lr_sched and e % 2 == 0: lr_sched.step()
    # if do_save and (e % 10 == 0 or e + 1 == num_epochs):
    if do_save and (e + 1 == num_epochs):
      torch.save(make_checkpoint(enc, dec), os.path.join(cp_p, f'{cp_name}-{e}.pth'))

def train_set(arg_sets):
  for i, args in enumerate(arg_sets):
    print(f'\t{i+1} of {len(arg_sets)}', flush=True)
    train(*args)

def eval_set(arg_sets):
  for args in arg_sets:
    hp, do_save_results, res_p, cp_name_split, eval_args = args
    out = evaluate_ticks(*eval_args)
    out['hp'] = hp
    if do_save_results: savemat(os.path.join(res_p, f'{cp_name_split}.mat'), out)

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
  res = torch.zeros((np, np))
  for i in range(np):
    for j in range(np):
      # res2[i, j] = i/100.0 * j/100.0
      # res2[-i, -j] = (i/100.0) * 0.5
      z = torch.Tensor([x[j], x[i]]).unsqueeze(0)
      expect_z = dec(z).detach()
      e = expect_z * expect_z.log()
      e[expect_z == 0.] = 0.
      res[i, j] = -torch.sum(e)
  return res

def discrete_entropy(ys, nc: int) -> float:
  fy = np.zeros((nc,))
  for i in range(nc): fy[i] = (ys == i).sum().item()
  fy = fy / fy.sum()
  hy = -(fy * np.log(fy))
  hy[fy == 0] = 0
  return hy.sum().item()

def evaluate_ticks(
  interface, enc: RecurrentEncoder, dec: Decoder, data_test, ctx: plotting.PlotContext):
  """"""
  samps = [x for x in data_test]
  ims = [x[0] for x in samps]
  ys = [x[1] for x in samps]
  ims = torch.concat(ims, dim=0)
  ys = torch.concat(ys, dim=0)

  # ims, ys = make_batch(data_test)
  batch_size = ims.shape[0]
  
  num_ticks = 16
  num_mc_z_eval = 12  # from VIB paper

  accs = np.zeros((num_ticks,))
  errs = np.zeros_like(accs)
  mc_accs = np.zeros_like(accs)
  mc_errs = np.zeros_like(accs)
  ticks = np.arange(num_ticks) + 1.
  pys = np.zeros((num_ticks, ys.shape[0], dec.nc))

  for t in range(num_ticks):
    print(f'{t+1} of {num_ticks}')
    _, res = interface.forward(enc, dec, ims, ys, {'num_ticks': t+1})
    _, mc_res = interface.forward(enc, dec, ims, ys, {'num_ticks': t+1}, num_z_eval=num_mc_z_eval)

    accs[t] = res['acc']
    errs[t] = res['err']
    pys[t] = res['yhat'].detach().cpu().squeeze(-1).numpy()
    mc_accs[t] = mc_res['acc']
    mc_errs[t] = mc_res['err']
  
  # evaluate with maximum ticks
  _, res = interface.forward(enc, dec, ims, ys, {'num_ticks': num_ticks})

  # summarize change in cov over ticks
  covs = np.zeros((3, num_ticks))
  areas = np.zeros((1, num_ticks))
  med_areas = np.zeros_like(areas)
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

    if enc.full_cov:
      cov = L @ L.transpose(-1, -2)
      covs[0, t] = cov[:, 0, 0].abs().mean()
      covs[1, t] = cov[:, -1, -1].abs().mean()
      covs[2, t] = cov[:, 1, 0].abs().mean()
      tmp_areas = np.zeros((cov.shape[0],))

      for i in range(cov.shape[0]):
        eigvals, eigvecs = np.linalg.eigh(cov.select(0, i).detach().cpu().numpy())
        width, height = 2 * np.sqrt(eigvals)
        mn, mx = min(width, height), max(width, height)
        tmp_areas[i] = np.pi * width * height
        areas[0, t] += np.pi * width * height
        anisos[0, t] += abs(1. - mx/mn)

      med_areas[0, t] = np.median(tmp_areas)
      areas[0, t] /= cov.shape[0]
      anisos[0, t] /= cov.shape[0]

  if ctx.show_plot or ctx.do_save:
    plotting.plot_line(ticks, errs * 100., 'Ticks', 'Error (%)', context=ctx)
    plotting.plot_line(ticks, mis[0, :], 'Ticks', 'I(X; Z)', context=ctx)
    plotting.plot_line(ticks, mis[1, :], 'Ticks', 'I(Y; Z)', context=ctx)
    plotting.plot_lines(ticks, covs.T, ['z_11', 'z_22', 'z_12'], 'Ticks', 'Covariances', context=ctx)
    plotting.plot_line(ticks, areas.T, 'Ticks', 'Area of ellipse', context=ctx)
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

  out = {
    'ticks': ticks,
    'errs': errs,
    'mc_errs': mc_errs,
    'mis': mis,
    'covs': covs,
    'areas': areas,
    'anisos': anisos,
    'py': pys,
    'y': ys.detach().cpu().numpy()
  }

  return out

def make_checkpoint(enc, dec):
  cp = {
    'enc_state': enc.state_dict(), 'dec_state': dec.state_dict(), 
    'enc_params': enc.ctor_params(), 'dec_params': dec.ctor_params()}
  return cp

def instantiate_model(cp: Dict, enc_fn):
  if 'input_dim' not in cp['enc_params']: cp['enc_params']['input_dim'] = 784
  if 'hidden_dim' not in cp['enc_params']: cp['enc_params']['hidden_dim'] = 1024
  if 'rnn_cell_type' not in cp['enc_params']: cp['enc_params']['rnn_cell_type'] = 'rnn'
  enc = enc_fn(**cp['enc_params']); enc.load_state_dict(cp['enc_state'])
  dec = Decoder(**cp['dec_params']); dec.load_state_dict(cp['dec_state'])
  return enc, dec

def split_array_indices(M: int, N: int) -> List[np.ndarray[int]]:
  # Return indices to split sequence with length `M` into at most `N` disjoint sets
  if M >= N:
    subset_size = M // N
    subsets = []
    for i in range(N):
      start = i * subset_size
      end = (i + 1) * subset_size if i + 1 < N else M
      subsets.append(np.array(range(start, end)))
    return subsets
  else:
    # Not enough elements to form N subsets; return one-element subsets
    return [np.array(range(i, i+1)) for i in range(M)]

# ------------------------------------------------------------------------------------------------

def main():
  is_cluster = True
  num_processes = 4 if is_cluster else 5
  # num_processes = 0
  task_type = 'mnist'
  # task_type = 'logic'
  base_p = '/scratch/naf264/explore-variational-rnn' if is_cluster else os.getcwd()
  do_train = True
  rand_ticks = True
  do_save_results = True
  num_epochs = 100
  batch_size = 100 if task_type == 'mnist' else 1024

  if task_type == 'mnist':
    # K = 2
    # full_cov = True
    K = 256
    full_cov = False
  elif task_type == 'logic':
    K = 2
    full_cov = False
  elif task_type == 'parity':
    K = 4
    full_cov = False
  else:
    assert False, f'Unhandled task type: "{task_type}"'
  
  # enc_hds = [32, 64, 128, 256, 512, 1024]
  # betas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.]
  # max_num_ticks_set = [6]

  # enc_hds = [1024]
  # betas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.]
  # betas = [1e-4, 1e-3, 1e-2, 1e-1]
  # betas = [0.]
  # max_num_ticks_set = [x + 1 for x in range(16)]
  # max_num_ticks_set = [2, 4, 6, 8]
  seeds = [x + 61 for x in range(5)]

  # debug logic task
  # seeds = seeds[:1]
  # enc_hds = [512]
  # max_num_ticks_set = [10]
  # betas = [0.]

  # debug original mnist
  enc_hds = [1024]
  # betas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
  # betas = [1e-9, 1e-8, 1e-7]
  betas = list(np.logspace(-2, -1, 16))
  # max_num_ticks_set = [1, 2, 4, 6, 8]
  max_num_ticks_set = [1, 2, 4, 6, 8]

  nb = 1 if do_train else len(betas)
  ns = 1 if do_train else len(max_num_ticks_set)
  nss = 1 if do_train else len(seeds)

  # loading all of the model combinations for evaluation is too memory intensive, do it in batches.
  bi = split_array_indices(len(betas), nb)
  si = split_array_indices(len(max_num_ticks_set), ns)
  ssi = split_array_indices(len(seeds), nss)
  ei = [*itertools.product(bi, si, ssi)]

  for i, e in enumerate(ei):
    print(f'{i+1} of {len(ei)}', flush=True)

    b, s, seed_ = e
    bs = [betas[b] for b in b]
    ss = [max_num_ticks_set[s] for s in s]
    seed_ = [seeds[s] for s in seed_]

    set_fn, arg_sets = prepare(
      do_train=do_train, betas=bs, enc_hds=enc_hds, max_num_ticks_set=ss,
      rand_ticks=rand_ticks, num_epochs=num_epochs, batch_size=batch_size, 
      task_type=task_type, do_save_results=do_save_results, base_p=base_p, seeds=seed_,
      K=K, full_cov=full_cov)

    run(set_fn, arg_sets, num_processes)

# ------------------------------------------------------------------------------------------------

def run(set_fn, arg_sets, num_processes: int):
  if num_processes <= 0:
    set_fn(arg_sets)
  else:
    pi = split_array_indices(len(arg_sets), num_processes)
    process_args = [[arg_sets[x] for x in y] for y in pi]
    processes = [Process(target=set_fn, args=(args,)) for args in process_args]
    for p in processes: p.start()
    for p in processes: p.join()

def prepare(
  *, do_train: bool, betas: List[float], enc_hds: List[int], 
  max_num_ticks_set: List[int], rand_ticks: bool, num_epochs: int, 
  batch_size: int, task_type: str, do_save_results: bool, base_p: str, seeds: List[int],
  K: int, full_cov: bool):
  """
  """
  root_p = os.path.join(base_p, 'data')
  res_p = os.path.join(base_p, 'results')

  if not os.path.exists(root_p): os.makedirs(root_p, exist_ok=True)
  if not os.path.exists(res_p): os.makedirs(res_p, exist_ok=True)
  
  assert task_type in ['parity', 'logic', 'mnist']
  do_save_plots = False
  do_show_plots = False
  is_recurrent = True
  eval_epochs = [*np.arange(0, num_epochs, 10)] + [num_epochs-1]
  use_lr_sched = True
  lr = 1e-4
  # eval_epochs = eval_epochs[-1:]
  
  if task_type == 'logic':
    use_lr_sched = False
    lr = 1e-3
    rnn_cell_type = 'rnn'
    nc = 2
    num_ops = 10
    input_dim = num_ops * 10 + 2
    seq_len = 1
    ldp = {'batch_size': batch_size, 'seq_len': seq_len, 'num_ops': num_ops}
    data_train = DataLoader(LogicDataset(**ldp, num_samples=10_000), batch_size=batch_size)
    data_test = DataLoader(LogicDataset(**ldp, num_samples=10_000), batch_size=batch_size)

  elif task_type == 'parity':
    rnn_cell_type = 'rnn'
    nc = 2
    input_dim = 64
    nc = 2
    data_train = DataLoader(ParityDataset(60_000, input_dim), batch_size=batch_size)
    data_test = DataLoader(ParityDataset(10_000, input_dim), batch_size=batch_size)

  elif task_type == 'mnist':
    rnn_cell_type = 'rnn'
    nc = 10
    input_dim = 784
    data_train = make_mnist_dataloader(
      datasets.mnist.MNIST(root_p, download=True, train=True, transform=ToTensor()), batch_size)
    data_test = make_mnist_dataloader(
      datasets.mnist.MNIST(root_p, download=True, train=False, transform=ToTensor()), batch_size)

  p = [*itertools.product(betas, max_num_ticks_set, enc_hds, seeds, [0] if do_train else eval_epochs)]

  arg_sets = []
  for ip, cmb in enumerate(p):
    print(f'{ip+1} of {len(p)}', flush=True)

    beta, max_num_ticks, enc_hd, seed, epoch = cmb

    hp = {}
    hp['task_type'] = task_type
    hp['beta'] = beta
    K = hp['K'] = K
    # K = 256
    full_cov = hp['full_cov'] = full_cov
    hp['max_num_ticks'] = max_num_ticks
    hp['epoch'] = epoch
    hp['encoder_hidden_dim'] = enc_hd
    hp['rand_ticks'] = rand_ticks
    hp['seed'] = seed
    beta_str = f'{beta:0.4f}' if beta >= 1e-4 else str(beta)

    cp_name = f'checkpoint-beta_{beta_str}-full_cov_{full_cov}-recurrent_{is_recurrent}' + \
              f'-rand_ticks_{rand_ticks}-max_ticks_{max_num_ticks}-task_type_{task_type}'
    if enc_hd != 1024: cp_name = f'{cp_name}-enc_hd_{enc_hd}'
    if seed != _NO_RANDOM_SEED: cp_name = f'{cp_name}-seed_{seed}'
    if rnn_cell_type != 'rnn': cp_name = f'{cp_name}-rnn_{rnn_cell_type}'
    if do_train:
      cp_p = os.path.join(root_p, f'{cp_name}.pth')
    else:
      cp_name = f'{cp_name}-{epoch}'
      cp_p = os.path.join(root_p, f'{cp_name}.pth')
      if not os.path.exists(cp_p): print(f'No such file: {cp_name}'); continue

    enc_fn = RecurrentEncoder
    interface = ModelInterface(
      forward=forward_recurrent,
      encoder_params=RecurrentEncoderParams(max_num_ticks) if is_recurrent else EncoderParams(),
      random_seed=seed
    )

    if do_train:
      enc = enc_fn(
        K=K, full_cov=full_cov, input_dim=input_dim, hidden_dim=enc_hd, beta=beta, 
        rnn_cell_type=rnn_cell_type)
      dec = Decoder(K=K, nc=nc)
    else:
      cp = torch.load(cp_p)
      enc, dec = instantiate_model(cp, enc_fn)

    print(f'Enc params: {count_parameters(enc)} | Dec params: {count_parameters(dec)}')

    if do_train:
      arg_sets.append((
        interface, enc, dec, num_epochs, data_train, data_test, root_p, cp_name, do_save_results,
        use_lr_sched, lr
      ))
    else:
      for id, ds in enumerate([data_test]):
        sn = 'test' if id == 0 else 'train'
        hp['split'] = sn
        cp_name_split = cp_name if sn == 'test' else f'train_{cp_name}'
        ctx = plotting.PlotContext()
        ctx.subdir = cp_name_split
        ctx.do_save = do_save_plots
        ctx.show_plot = do_show_plots
        arg_sets.append(
          (hp.copy(), do_save_results, res_p, cp_name_split, (interface, enc, dec, ds, ctx))
        )
  
  set_fn = train_set if do_train else eval_set
  return set_fn, arg_sets

# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':
  main()