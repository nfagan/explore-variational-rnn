import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.io import savemat
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from tasks import generate_logic_task_sequence, generate_addition_task_sequence

import os, random, string
from itertools import product
from multiprocessing import Process
from typing import List
from dataclasses import dataclass

# ------------------------------------------------------------------------------------------------

@dataclass
class EvaluateContext:
  save_p: str
  do_save: bool

class AdditionDataset(Dataset):
  def __init__(self, *, seq_len: int, max_num_digits: int, batch_size: int):
    super().__init__()
    self.seq_len = seq_len
    self.max_num_digits = max_num_digits
    self.batch_size = batch_size

  def __len__(self):
    return self.batch_size
  
  def __getitem__(self, idx):
    x, y, m = generate_addition_task_sequence(self.seq_len, self.max_num_digits)
    x = x.T
    return x, y.T

class LogicDataset(Dataset):
  def __init__(
    self, batch_size: int, seq_len: int, num_ops: int, num_samples: int = None, fixed_num_ops: int = None):
    """"""
    super().__init__()
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.num_ops = num_ops
    self.samples = []
    self.num_samples = num_samples
    self.fixed_num_ops = fixed_num_ops

    if num_samples is not None:
      for _ in range(num_samples):
        x, y, _ = generate_logic_task_sequence(self.seq_len, self.num_ops, self.fixed_num_ops)
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
      x, y, intermediates = generate_logic_task_sequence(self.seq_len, self.num_ops, self.fixed_num_ops)
    # x: (seq_len x N) | y: (seq_len x 1) | m: (seq_len x 1)
    y = y.type(torch.long)
    x = x.T
    y = y.T.squeeze(0)
    # x: (N x seq_len) | y: (seq_len,)
    return x, y, intermediates

# ------------------------------------------------------------------------------------------------

class RecurrentClassifier(nn.Module):
  def __init__(
    self, *, input_dim: int, hidden_dim: int, rnn_cell_type: str, nc: int, bottleneck_K: int):
    
    assert rnn_cell_type in ['rnn', 'lstm']
    super().__init__()

    if rnn_cell_type == 'rnn':    self.rnn = nn.RNNCell(input_dim, hidden_dim)
    elif rnn_cell_type == 'lstm': self.rnn = nn.LSTMCell(input_dim, hidden_dim)
    else: assert False

    if bottleneck_K is not None:
      # uses a bottleneck
      num_distribution_params = 2 * bottleneck_K
      self.encoder_params = nn.Linear(hidden_dim, num_distribution_params)
      self.dec = nn.Linear(bottleneck_K, nc)
      self.has_bottleneck = True
      self.K = bottleneck_K
    else:
      self.dec = nn.Linear(hidden_dim, nc)
      self.has_bottleneck = False
      self.K = None

    self.halt = nn.Linear(hidden_dim, 1)
    
    # self.bottleneck = nn.Linear(hidden)
    self.rnn_cell_type = rnn_cell_type
    self.nc = nc

  def compute_y(self, h: torch.Tensor):
    if self.has_bottleneck:
      fe = self.encoder_params(h)
      mus = fe[:, :self.K]
      sigmas = F.softplus(fe[:, self.K:] - 5.)
      eps = torch.randn_like(sigmas)
      z = eps * sigmas + mus
      y = self.dec(z)
      return y, z, mus, sigmas
    else:
      y = self.dec(h)
      return y, None, None, None

  def fixed_ticks_step(self, sx: torch.Tensor, xi: torch.Tensor, num_ticks: int):
    sn = sx
    hs = []
    for _ in range(num_ticks):
      sn = self.rnn(xi, sn)
      h = sn[0] if self.rnn_cell_type == 'lstm' else sn
      y, z, mus, sigmas = self.compute_y(h)
      hs.append(h)

    pt = torch.zeros((xi.shape[0],))
    yt = y
    sx = sn
    nt = torch.ones((xi.shape[0],), dtype=torch.int64) * num_ticks
    hs = torch.stack(hs, dim=-1)

    if self.has_bottleneck:
      addtl = {'z': z, 'mus': mus, 'sigmas': sigmas, 'h': hs}
    else:
      addtl = {'h': hs}

    return sx, yt, pt, nt, addtl
  
  def non_mean_field_act_step(self, sx: torch.Tensor, xi: torch.Tensor, eps_halting: float, M: int):
    sn = sx
    pt = 0. # cumuluative halting probability
    p_hist = []
    keep_processing = torch.ones((xi.shape[0],), dtype=bool)
    n = 0

    yt = torch.zeros((xi.shape[0], self.nc))
    while n < M and keep_processing.any():
      sn0 = sn
      sn = self.rnn(xi, sn)

      h = sn[0]
      pn = torch.sigmoid(self.halt(h)).squeeze(1)

      if sn0 is not None:
        state0 = list(sn0) if self.rnn_cell_type == 'lstm' else [sn0]
        state1 = list(sn) if self.rnn_cell_type == 'lstm' else [sn]
        for si in range(len(state1)):
          t = (pn * keep_processing)[:, None].tile(1, state1[si].shape[1])
          state1[si] = torch.lerp(state0[si], state1[si], t)
        sn = tuple(state1)

      pt += pn
      newly_finished = pt >= 1. - eps_halting
      p_hist.append(pn)
      keep_processing[newly_finished] = False
      n += 1

    yt, z, mus, sigmas = self.compute_y(h)

    p_hist = torch.stack(p_hist, dim=-1)
    p_max = torch.min(torch.tensor(1.), torch.cumsum(p_hist, dim=-1))
    p_max[:, -1] = 1.
    # determine the number of steps chosen per input
    nt = torch.argmax(p_max, dim=1, keepdim=True).squeeze(1)
    rt = 1. - p_max[torch.arange(p_max.shape[0]), nt-1]
    pt = nt + rt

    if self.has_bottleneck:
      addtl = {'z': z, 'mus': mus, 'sigmas': sigmas}
    else:
      addtl = {}

    return sx, yt, pt, nt, addtl

  def act_step(self, sx: torch.Tensor, xi: torch.Tensor, eps_halting: float, M: int, fixed_M: bool):
    sn = sx
    pt = 0. # cumuluative halting probability
    p_hist = []
    y_hist = []
    s_hist = []
    c_hist = []
    keep_processing = torch.ones((xi.shape[0],), dtype=bool)
    n = 0
    while n < M:
      if (not fixed_M) and (not keep_processing.any()): break
      sn = self.rnn(xi, sn)
      h = sn[0] if self.rnn_cell_type == 'lstm' else sn
      pn = torch.sigmoid(self.halt(h)).squeeze(1)
      y, z, mus, sigmas = self.compute_y(h)
      pt += pn
      p_hist.append(pn)
      s_hist.append(h)
      if self.rnn_cell_type == 'lstm': c_hist.append(sn[1])
      y_hist.append(y)
      keep_processing[pt >= 1. - eps_halting] = False
      n += 1

    p_hist = torch.stack(p_hist, dim=-1)
    y_hist = torch.stack(y_hist, dim=-1)
    s_hist = torch.stack(s_hist, dim=-1)
    c_hist = torch.stack(c_hist, dim=-1)

    p_max = torch.min(torch.tensor(1.), torch.cumsum(p_hist, dim=-1))
    p_max[:, -1] = 1.
    ph = torch.cat([p_max[:, 0][:, None], torch.diff(p_max, dim=-1)], dim=-1)
    # weight states and outputs by halting probabilities
    st = torch.sum(ph.unsqueeze(1) * s_hist, dim=-1)
    if self.rnn_cell_type == 'lstm': ct = torch.sum(ph.unsqueeze(1) * c_hist, dim=-1)
    yt = torch.sum((ph.unsqueeze(-1) * y_hist.transpose(-1, -2)).transpose(-1, -2), dim=-1)
    # determine the number of steps chosen per input
    nt = torch.argmax(p_max, dim=1, keepdim=True).squeeze(1)
    rt = 1. - p_max[torch.arange(p_max.shape[0]), nt-1]
    # begin from the mean-field states for the next time-step
    if self.rnn_cell_type == 'lstm': sx = (st, ct)
    else: sx = st
    pt = nt + rt

    if self.has_bottleneck:
      addtl = {'z': z, 'mus': mus, 'sigmas': sigmas, 'h': s_hist}
    else:
      addtl = {'h': s_hist}

    return sx, yt, pt, nt, addtl

  def forward(
    self, x: torch.Tensor, sx: torch.Tensor = None, eps_halting = 1e-2, M = 14, 
    step_type: str = 'act', num_fixed_ticks: int = 6):
    assert step_type in ['act', 'fixed', 'fixed-act', 'non-mf-act']

    T = x.shape[-1]
    P = []
    Y = []
    N = []
    Addtl = {}

    for t in range(T):
      xi = x.select(-1, t)

      if step_type == 'act' or step_type == 'fixed-act':
        fixed_M = step_type == 'fixed-act'
        use_M = num_fixed_ticks if fixed_M else M
        sx, yt, pt, nt, addtl = self.act_step(sx, xi, eps_halting, use_M, fixed_M)
      elif step_type == 'non-mf-act':
        sx, yt, pt, nt, addtl = self.non_mean_field_act_step(sx, xi, eps_halting, M)
      elif step_type == 'fixed':
        sx, yt, pt, nt, addtl = self.fixed_ticks_step(sx, xi, num_fixed_ticks)
      else: assert False

      if t == 0: Addtl = {k: [] for v, k in enumerate(addtl)}
      for k in addtl: Addtl[k].append(addtl[k])

      # append to sequence
      P.append(pt)
      Y.append(yt)
      N.append(nt)

    P = torch.stack(P, dim=-1)
    Y = torch.stack(Y, dim=-1)
    N = torch.stack(N, dim=-1)
    P = torch.sum(P, dim=-1)
    for k in Addtl: 
      if k != 'h': Addtl[k] = torch.stack(Addtl[k], dim=-1)
    return Y, P, N, Addtl

# ------------------------------------------------------------------------------------------------

def make_cp_id(hps: dict):
  cp_id = str(hps).replace("'", '').replace(': ', '_')
  cp_id = cp_id.replace('{', '').replace('}', '').replace(', ', '-')
  return cp_id

def sequence_acc(ys: torch.Tensor, yh: torch.Tensor):
  return torch.sum(yh == ys) / ys.numel()

def sequence_error_rate(ys: torch.Tensor, yh: torch.Tensor):
  corr = yh == ys
  err = torch.any(~corr, dim=1)
  err_rate = torch.mean(err.type(torch.float))
  return err_rate

def train(
  enc: RecurrentClassifier, foward_fn, data_train: DataLoader, 
  loss_fn, epoch_cb, num_epochs: int, lr: float, random_seed: int):
  """"""
  torch.manual_seed(random_seed)
  optim = torch.optim.Adam([*enc.parameters()], lr=lr)

  for e in range(num_epochs):
    for xs, ys, intermediates in data_train:
      y, p, n, addtl = foward_fn(enc, xs)
      err_rate = sequence_error_rate(ys, torch.argmax(torch.softmax(y, dim=1), dim=1))

      optim.zero_grad()
      L = loss_fn(y, ys, p, addtl)
      L.backward()
      optim.step()

      nf = n.type(torch.float)
      mu_n = torch.mean(nf)
      max_n = torch.max(nf)

      print(
        f'{e+1} of {num_epochs} | Loss: {L.item():.3f}' + 
        f' | N: {mu_n.item():.2f} (max {max_n.item()}) | Err: {err_rate.item():.3f}', flush=True
      )
      
    epoch_cb(enc, e)

def do_evaluate(enc: RecurrentClassifier, foward_fn, data: DataLoader, loss_fn):
  for xs, ys, intermediates in data:
    y, p, n, addtl = foward_fn(enc, xs)
    addtl['n'] = n
    addtl['p'] = p
    addtl['xs'] = xs
    addtl['ys'] = ys
    addtl['intermediates'] = intermediates
    yh = torch.argmax(torch.softmax(y, dim=1), dim=1)
    err_rate = sequence_error_rate(ys, yh)
    seq_acc = sequence_acc(ys, yh)
    L = loss_fn(y, ys, p, addtl)
    return {'acc': seq_acc.item(), 'err_rate': err_rate.item(), 'ticks': n.float().mean().item()}, addtl
  
def decode_internal_reprs_logic_task(enc: RecurrentClassifier, foward_fn, loss_fn):
  data_prep = prepare_logic_task(batch_size=10_000, num_ops=10, fixed_num_ops=None)
  _, addtl = do_evaluate(enc, foward_fn, data_prep, loss_fn)
  last_ticki = addtl['n'][:, -1]  # last step of sequence
  last_h = addtl['h'][-1]
  last_h = last_h[torch.arange(last_h.shape[0]), :, last_ticki]

  num_ops = addtl['intermediates'][:, :, -1, :][:, -1, -1]
  op_indices = addtl['intermediates'][:, :, 0, :]
  op_indices = op_indices[torch.arange(op_indices.shape[0]), num_ops-1, -1]

  clfs = []
  clf_ys = [addtl['ys'][:, -1].detach().cpu().numpy(), op_indices.detach().cpu().numpy()]
  for clf_Y in clf_ys:
    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    clf_X = last_h.detach().cpu().numpy()
    clf.fit(clf_X, clf_Y)
    clfs.append(clf)

  # now evaluate over ticks
  data_prep = prepare_logic_task(batch_size=1_000, num_ops=10, fixed_num_ops=None)
  _, addtl = do_evaluate(enc, foward_fn, data_prep, loss_fn)
  hs = addtl['h']
  nt = max([x.shape[-1] for x in hs])

  class_pred_ys = []
  for clf in clfs:
    class_preds = []
    for si in range(len(hs)):
      hi = hs[si]
      class_pred = torch.ones(hi.shape[0], nt) * -1
      for pi in range(hi.shape[2]):
        hp = hi[:, :, pi].detach().cpu().numpy()
        yp = clf.predict(hp)
        class_pred[:, pi] = torch.tensor(yp)
      class_preds.append(class_pred.detach().cpu().numpy())
    class_pred_ys.append(np.stack(class_preds, axis=-1))
  
  class_pred_ys = np.stack(class_pred_ys, axis=-1)
  ints = addtl['intermediates']
  rd = {
    'n': addtl['n'].detach().cpu().numpy(), 
    'pred_ys': class_pred_ys, 'intermediates': ints.detach().cpu().numpy()}
  return rd
  
def evaluate(
  ctx: EvaluateContext, enc: RecurrentClassifier, foward_fn, data: DataLoader, 
  loss_fn, train_epoch: int, hps: dict):
  """"""
  hps = {**hps}
  hps['num_fixed_ticks'] = -1
  hps['seq_len'] = 3

  if True:
    # evaluate generalization performance when sequences are longer than those seen during training
    if hps['task_type'] == 'logic':
      for i in range(3):
        slen = 4 + i
        hp_res = {**hps}
        hp_res['seq_len'] = slen
        data_prep = prepare_logic_task(batch_size=10_000, num_ops=10, fixed_num_ops=None, seq_len=slen)
        evaluate_(ctx, enc, foward_fn, data_prep, loss_fn, train_epoch, hp_res)

  if False:
    # decode hidden state representation(s) of output
    if hps['task_type'] == 'logic':
      hp_res = {**hps}
      decode_res = decode_internal_reprs_logic_task(enc, foward_fn, loss_fn)
      decode_res['hps'] = hp_res
      if ctx.do_save:
        matname = GenCPFnameFn(hp_res, True)(train_epoch).replace('.pth', '.mat')
        savemat(os.path.join(ctx.save_p, 'decoding', matname), decode_res)

  if False:
    # evaluate performance when varying example difficulty
    if hps['task_type'] == 'logic':
      num_ops = [*np.arange(0, 10, 2)] + [9]
      for i in num_ops:
        print(f'{i+1} of {len(num_ops)}')
        data_prep = prepare_logic_task(batch_size=10_000, num_ops=10, fixed_num_ops=i+1)
        hps_fixed = {**hps}
        hps_fixed['fixed_num_ops'] = i + 1
        evaluate_(ctx, enc, foward_fn, data_prep, loss_fn, train_epoch, hps_fixed)

  if False:
    # (default) evaluate performance for classifying `data`
    hps_dflt = {**hps}
    evaluate_(ctx, enc, foward_fn, data, loss_fn, train_epoch, hps_dflt)

def evaluate_(
  ctx: EvaluateContext, enc: RecurrentClassifier, foward_fn, data: DataLoader, 
  loss_fn, train_epoch: int, hps: dict):
  """"""
  hp_res = {**hps}
  hp_res['num_fixed_ticks'] = -1
  res, _ = do_evaluate(enc, foward_fn, data, loss_fn)
  res['hps'] = hp_res
  if ctx.do_save:
    savemat(os.path.join(ctx.save_p, GenCPFnameFn(hp_res, True)(train_epoch).replace('.pth', '.mat')), res)

  for t in [*np.arange(0, 16, 2)]:
    num_fixed_ticks = t + 1
    for st in ['fixed', 'fixed-act']:
      forward_fixed_fn = lambda enc, xs: enc(xs, step_type=st, num_fixed_ticks=num_fixed_ticks)
      res, addtl = do_evaluate(enc, forward_fixed_fn, data, loss_fn)
      hp_res = {**hps}
      hp_res['step_type'] = st
      hp_res['num_fixed_ticks'] = num_fixed_ticks
      save_fname = GenCPFnameFn(hp_res, True)(train_epoch).replace('.pth', '.mat')
      res['hps'] = hp_res
      if ctx.do_save:
        savemat(os.path.join(ctx.save_p, save_fname.replace('.pth', '.mat')), res)

# ------------------------------------------------------------------------------------------------

def prepare_logic_task(*, batch_size: int, num_ops: int, fixed_num_ops: int, seq_len: int = 3):
  ds = LogicDataset(
    batch_size=batch_size, seq_len=seq_len, num_ops=num_ops, 
    num_samples=None, fixed_num_ops=fixed_num_ops)
  return DataLoader(ds, batch_size=batch_size)

def prepare_data(task_type: str, batch_size: int):
  if task_type == 'logic':
    nc = 2
    num_ops = 10
    input_dim = num_ops * 10 + 2
    stride = 1
    data_train = prepare_logic_task(batch_size=batch_size, num_ops=num_ops, fixed_num_ops=None)

  elif task_type == 'addition':
    max_num_digits = 5
    input_dim = max_num_digits * 10
    stride = 11
    nc = stride * (max_num_digits + 1)
    ldp = {'batch_size': batch_size, 'seq_len': 5, 'max_num_digits': max_num_digits}
    data_train = DataLoader(AdditionDataset(**ldp), batch_size=batch_size)

  else: assert False

  return data_train, input_dim, nc

def kl_full_gaussian(mu: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
  k = mu.size(-1)
  # trace(Σ) = ‖L‖_F² because trace(L Lᵀ) = Σᵢⱼ L_{ij}²
  trace_term = (L ** 2).sum(dim=(-2, -1))
  # log|Σ| = 2·Σ_i log L_ii (diagonal of a Cholesky factor is > 0)
  log_det = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
  quad_term = mu.pow(2).sum(-1) # μᵀμ
  kl = 0.5 * (trace_term + quad_term - k - log_det)
  return kl.sum()

def kl_div(mus: torch.Tensor, sigs: torch.Tensor):
  for i in range(mus.shape[-1]):
    mu, sig = mus.select(-1, i), sigs.select(-1, i)
    kl = kl_full_gaussian(mu, sig)
    s = kl if i == 0 else s + kl
  return s / mus.shape[-1]

class TrainCBFn(object):
  def __init__(self, *, do_save: bool, cp_save_interval: int, gen_cp_fname_fn, hps: dict):
    self.do_save = do_save
    self.cp_save_interval = cp_save_interval
    self.gen_cp_fname_fn = gen_cp_fname_fn
    self.hps = hps

  def __call__(self, enc, e: int):
    if not self.do_save: return
    if e % self.cp_save_interval != 0: return
    cp_fname = self.gen_cp_fname_fn(e)
    cp = {'state': enc.state_dict(), 'hps': self.hps}
    torch.save(cp, os.path.join(os.getcwd(), 'data', cp_fname))

class GenCPFnameFn(object):
  def __init__(self, hps: dict, eval_mode: bool):
    self.hps = hps
    self.eval_mode = eval_mode
  def __call__(self, e: int):
    res = f'act-checkpoint-{make_cp_id(self.hps)}-{e}.pth'
    if self.eval_mode:
      res = res[:min(len(res), 128)].replace('.pth', '')
      res += ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
      res = f'{res}.pth'
    return res

class ForwardFn(object):
  def __init__(self, *, step_type: str, num_fixed_ticks: int, M: int):
    self.step_type = step_type
    self.num_fixed_ticks = num_fixed_ticks
    self.M = M

  def __call__(self, enc: RecurrentClassifier, xs: torch.Tensor): 
    return enc(xs, step_type=self.step_type, num_fixed_ticks=self.num_fixed_ticks, M=self.M)

class LossFn(object):
  def __init__(self, *, ponder_cost: float, beta: float, weight_normalization_type: str):
    assert weight_normalization_type in ['norm', 'none']
    self.ponder_cost = ponder_cost
    self.beta = beta
    self.weight_normalization_type = weight_normalization_type

  def __call__(self, y, ys, p, addtl: dict):
    L_acc = nn.functional.cross_entropy(y, ys)
    L_ponder = torch.mean(p)
    L_kl = 0.
    if 'mus' in addtl: L_kl = kl_div(addtl['mus'], addtl['sigmas']) / y.shape[0]
    weights = np.array([1., self.ponder_cost, self.beta])
    if self.weight_normalization_type == 'norm': 
      weights /= np.linalg.norm(weights)
    elif self.weight_normalization_type == 'none':
      pass
    else: assert False
    # return L_acc + ponder_cost*L_ponder + beta*L_kl
    return weights[0]*L_acc + weights[1]*L_ponder + weights[2]*L_kl

def prepare(
  *, ponder_cost: float, do_save: bool, step_type: str, num_fixed_ticks: int, hps: dict,
  data_train: DataLoader, input_dim: int, nc: int, eval_epoch: int, bottleneck_K: int, beta: float, 
  M: int, weight_normalization_type: str):
  """"""
  cp_save_interval = 2000
  rnn_hidden_dim = hps['rnn_hidden_dim'] = 512
  rnn_cell_type = hps['rnn_cell_type'] = 'lstm'

  hps['ponder_cost'] = ponder_cost
  hps['num_fixed_ticks'] = num_fixed_ticks
  hps['step_type'] = step_type
  # num_classif = nc // stride

  enc = RecurrentClassifier(
    input_dim=input_dim, hidden_dim=rnn_hidden_dim,
    rnn_cell_type=rnn_cell_type, nc=nc, bottleneck_K=bottleneck_K
  )
  
  gen_cp_fname_fn = GenCPFnameFn(hps, False)

  if eval_epoch is not None:
    enc.eval()
    cp = torch.load(os.path.join(os.getcwd(), 'data', gen_cp_fname_fn(eval_epoch)))
    enc.load_state_dict(cp['state'])

  train_cb_fn = TrainCBFn(
    do_save=do_save, cp_save_interval=cp_save_interval, gen_cp_fname_fn=gen_cp_fname_fn, hps=hps)
  forward_fn = ForwardFn(step_type=step_type, num_fixed_ticks=num_fixed_ticks, M=M)
  loss_fn = LossFn(ponder_cost=ponder_cost, beta=beta, weight_normalization_type=weight_normalization_type)

  args = (enc, forward_fn, data_train, loss_fn, train_cb_fn, gen_cp_fname_fn)
  return args

# ------------------------------------------------------------------------------------------------

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
  
def train_set(arg_sets): 
  for args in arg_sets: train(*args)
def eval_set(arg_sets):
  for args in arg_sets: evaluate(*args)

def run(set_fn, arg_sets, num_processes: int):
  if num_processes <= 0:
    set_fn(arg_sets)
  else:
    pi = split_array_indices(len(arg_sets), num_processes)
    process_args = [[arg_sets[x] for x in y] for y in pi]
    processes = [Process(target=set_fn, args=(args,)) for args in process_args]
    for p in processes: p.start()
    for p in processes: p.join()

def main():
  eval_batch_size = 10_000
  num_train_epochs = 30_001
  do_save = True
  num_processes = 0

  M = 20
  weight_normalization_type = 'none'
  # weight_normalization_type = 'norm'
  seeds = [61, 62, 63, 64, 65]
  eval_epochs = list(np.arange(0, num_train_epochs, 2_000))
  # ponder_costs = [1e-3 * 0.5, 1e-3 * 1, 1e-3 * 2, 1e-3 * 3, 1e-3 * 4]
  # ponder_costs = [1e-3 * 2, 1e-3 * 3, 1e-2]
  ponder_costs = [1e-3 * 1]
  bottleneck_Ks = [512]

  betas = [0, 1e-2, 1e-1, 2e-2, 3e-2]
  # betas = [1e-2*(1/2), 1e-2*(1/3), 1e-2*(1/4)]
  # betas = [1e-2*(1/6), 1e-2*(1/8), 1e-2*(1/10)]

  # --- just one
  eval_epochs = [ eval_epochs[-1] ]
  # seeds = [ seeds[0] ]
  # ponder_costs = [ ponder_costs[0] ]
  # eval_epochs = [ eval_epochs[0] ]
  # --- just one

  # eval_epochs = [None]
  # ponder_costs = [1e-3 * 1]

  its = [*product(seeds, eval_epochs, ponder_costs, betas, bottleneck_Ks)]
  arg_sets = []

  for i, it in enumerate(its):
    print(f'{i+1} of {len(its)}')

    seed, eval_epoch, ponder_cost, beta, bottleneck_K = it
    hps = {}

    task_type = hps['task_type'] = 'logic'
    train_batch_size = hps['batch_size'] = 32
    seed = hps['seed'] = seed
    step_type = 'act'
    # step_type = 'non-mf-act'
    num_fixed_ticks = 6
    lr = 1e-3
    hps['bottleneck_K'] = bottleneck_K
    hps['beta'] = beta
    hps['M'] = M
    hps['weight_normalization_type'] = weight_normalization_type

    batch_size = train_batch_size if eval_epoch is None else eval_batch_size
    data_train, input_dim, nc = prepare_data(task_type, batch_size)

    enc, forward_fn, data_train, loss_fn, train_cb_fn, gen_cp_fname_fn = prepare(
      ponder_cost=ponder_cost, do_save=do_save,
      step_type=step_type, num_fixed_ticks=num_fixed_ticks, hps=hps,
      data_train=data_train, nc=nc, input_dim=input_dim, eval_epoch=eval_epoch,
      bottleneck_K=bottleneck_K, beta=beta, M=M, weight_normalization_type=weight_normalization_type
    )

    if eval_epoch is not None: # evaluate
      hps['epoch'] = eval_epoch
      eval_ctx = EvaluateContext(save_p=os.path.join(os.getcwd(), 'results'), do_save=do_save)
      arg_sets.append((eval_ctx, enc, forward_fn, data_train, loss_fn, eval_epoch, hps))
    else:
      arg_sets.append((enc, forward_fn, data_train, loss_fn, train_cb_fn, num_train_epochs, lr, seed))

  set_fn = train_set if eval_epoch is None else eval_set
  run(set_fn, arg_sets, num_processes)

# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':
  main()