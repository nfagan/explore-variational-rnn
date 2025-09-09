import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.io import savemat
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from tasks import generate_logic_task_sequence, generate_addition_task_sequence, generate_parity_task_sequence

import os, random, string, datetime
from itertools import product
from multiprocessing import Process
from typing import List, Union
from dataclasses import dataclass

# ------------------------------------------------------------------------------------------------

@dataclass
class EvaluateContext:
  save_p: str
  do_save: bool
  default_batch_size: int

class ParityDataset(Dataset):
  def __init__(self, *, batch_size: int, vector_length: int):
    super().__init__()
    self.batch_size = batch_size
    self.vector_length = vector_length

  def __len__(self):
    return self.batch_size
  
  def __getitem__(self, idx):
    x, y, _ = generate_parity_task_sequence(self.vector_length)
    y = y.type(torch.long)
    x = x.reshape(self.vector_length, 1)
    intermediates = torch.tensor([0.])
    # x: (N x seq_len) | y: (seq_len,) | intermediates: (seq_len,)
    return x, y, intermediates

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
    self, batch_size: int, seq_len: int, num_ops: int, 
    num_samples: int = None, fixed_num_ops: int = None, fixed_num_ops_p = None, 
    sample_ops_with_replacement = True):
    """"""
    super().__init__()
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.num_ops = num_ops
    self.samples = []
    self.num_samples = num_samples
    self.fixed_num_ops = fixed_num_ops
    self.fixed_num_ops_p = fixed_num_ops_p
    self.sample_ops_with_replacement = sample_ops_with_replacement

    if fixed_num_ops_p is not None:
      assert fixed_num_ops is None, 'Only specify one of `fixed_num_ops` and `fixed_num_ops_p`'
      assert len(fixed_num_ops_p) == num_ops

    if num_samples is not None:
      for _ in range(num_samples):
        x, y, _ = generate_logic_task_sequence(
          self.seq_len, self.num_ops, self.fixed_num_ops, self.sample_ops_with_replacement)
        self.samples.append((x, y))
  
  def __len__(self):
    if self.num_samples is not None:
      return self.num_samples
    else:
      return self.batch_size
  
  def __getitem__(self, idx):
    if self.num_samples is not None:
      x, y = self.samples[idx]

    elif self.fixed_num_ops_p is not None:
      num_ops = 1 + np.random.choice(self.num_ops, size=(1,), replace=True, p=self.fixed_num_ops_p).item()
      x, y, intermediates = generate_logic_task_sequence(
        self.seq_len, self.num_ops, num_ops, self.sample_ops_with_replacement)

    else:
      x, y, intermediates = generate_logic_task_sequence(
        self.seq_len, self.num_ops, self.fixed_num_ops, self.sample_ops_with_replacement)

    # x: (seq_len x N) | y: (seq_len x 1) | m: (seq_len x 1)
    y = y.type(torch.long)
    x = x.T
    y = y.T.squeeze(0)
    # x: (N x seq_len) | y: (seq_len,)
    return x, y, intermediates

# ------------------------------------------------------------------------------------------------

class VRNNClassifier(nn.Module):
  def __init__(
    self, *, input_dim: int, hidden_dim: int, rnn_cell_type: str, nc: int, bottleneck_K: int):
    """"""
    # assert rnn_cell_type == 'rnn' # @TODO, enable other cell types
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.rnn_cell_type = rnn_cell_type
    self.nc = nc
    self.K = bottleneck_K
    self.encode_cell_state = rnn_cell_type == 'lstm' and True # and False
    self.enable_bottleneck = bottleneck_K is not None

    if self.enable_bottleneck:
      encode_mul = 2 if self.encode_cell_state else 1
      self.hidden_state_to_encoder_params = nn.Linear(hidden_dim * encode_mul, 2 * bottleneck_K)
      self.latent_to_hidden_state = nn.Linear(bottleneck_K, hidden_dim * encode_mul)

    self.hidden_state_to_output = nn.Linear(hidden_dim, nc)
    self.halt = nn.Linear(hidden_dim, 1)
    
    if rnn_cell_type == 'rnn': self.rnn = nn.RNNCell(input_dim, hidden_dim)
    elif rnn_cell_type == 'lstm': self.rnn = nn.LSTMCell(input_dim, hidden_dim)
    elif rnn_cell_type == 'gru': self.rnn = nn.GRUCell(input_dim, hidden_dim)
    else: assert False

  def extract_cell_states(self, sn):
    h = sn[0] if self.rnn_cell_type == 'lstm' else sn
    c = sn[1] if self.encode_cell_state else None
    return h, c

  def update_cell_state(self, h, c, sn):
    if self.rnn_cell_type == 'rnn' or self.rnn_cell_type == 'gru': sn = h
    elif self.encode_cell_state: sn = (h, c)
    else: sn = (h, sn[1])
    return sn

  def compute_latent_to_hidden_state(self, z: torch.Tensor):
    nonlin_on_h = True # make False to return to LSTM variant
    h = self.latent_to_hidden_state(z)
    if nonlin_on_h: h = torch.tanh(h)
    if self.encode_cell_state:
      c = h[:, self.hidden_dim:]
      if not nonlin_on_h: c = torch.tanh(c)
      h = h[:, :self.hidden_dim]
    else:
      c = None
    return h, c

  def compute_z(self, h: torch.Tensor, c: torch.Tensor = None):
    if c is not None: h = torch.cat([h, c], dim=-1)
    enc_p = self.hidden_state_to_encoder_params(h)
    mus = enc_p[:, :self.K]
    sigmas = F.softplus(enc_p[:, self.K:] - 5.)
    eps = torch.randn_like(sigmas)
    z = eps * sigmas + mus
    return z, mus, sigmas
  
  def fixed_ticks_step(self, sx: torch.Tensor, xi: torch.Tensor, num_ticks: int):
    """"""
    sn = sx
    hs = []
    
    for _ in range(num_ticks):
      sn = self.rnn(xi, sn)
      h, c = self.extract_cell_states(sn)
      z, mus, sigmas = self.compute_z(h, c)
      if self.enable_bottleneck: h, c = self.compute_latent_to_hidden_state(z)
      y = self.hidden_state_to_output(h)
      sn = self.update_cell_state(h, c, sn)
      hs.append(h)

    pt = torch.zeros((xi.shape[0],))
    yt = y
    sx = sn
    nt = torch.ones((xi.shape[0],), dtype=torch.int64) * num_ticks
    hs = torch.stack(hs, dim=-1)

    addtl = {'z': z, 'mus': mus, 'sigmas': sigmas, 'h': hs}

    return sx, yt, pt, nt, addtl

  def ponder_net_step(
    self, xi: torch.Tensor, sx: torch.Tensor, eps_halting: float, M: int, min_lambda_p: float):
    """
    """
    enable_halting = True
    if min_lambda_p is not None: min_lambda_p = torch.tensor(min_lambda_p)

    sn = sx
    p_hist = []
    y_hist = []
    z_hist = []
    m_hist = []
    s_hist = []
    h_hist = []

    nt = torch.zeros((xi.shape[0], 1), dtype=torch.long)
    yt = torch.zeros((xi.shape[0], self.nc))

    lambdas = torch.zeros((xi.shape[0], 1))
    keep_processing = torch.ones((xi.shape[0],), dtype=bool)
    eval_halted = torch.zeros_like(keep_processing)
    halted_at = torch.zeros(keep_processing.shape, dtype=torch.long)
    n = 0

    while n < M:
      if not keep_processing.any(): break
      
      sn = self.rnn(xi, sn)
      h, c = self.extract_cell_states(sn)
      
      # --- bottleneck
      if self.enable_bottleneck:
        z, mus, sigmas = self.compute_z(h, c)
        h, c = self.compute_latent_to_hidden_state(z)
        sn = self.update_cell_state(h, c, sn)

      # --- output & halting probability
      y = self.hidden_state_to_output(h)
      lambda_n = torch.sigmoid(self.halt(h))
      if min_lambda_p is not None: lambda_n = torch.maximum(lambda_n, min_lambda_p)
      # lambda_n = torch.clamp(lambda_n, torch.tensor(0.05), torch.tensor(1.-0.05))

      # --- evaluation: store outputs
      # when lambda_n is low, the probability of newly halting should be low      
      if enable_halting:
        eval_crit = torch.bernoulli(lambda_n).squeeze(1).type(torch.bool)
        eval_newly_halted = torch.logical_and(~eval_halted, eval_crit)
        eval_halted[eval_newly_halted] = True
        yt[eval_newly_halted, :] = y[eval_newly_halted, :]
        nt[eval_newly_halted] = n + 1
      # ---

      p_n1 = torch.prod(1. - lambdas, dim=1, keepdim=True)
      pn = lambda_n * p_n1
      lambdas = torch.cat([lambdas, lambda_n], dim=-1)
      p_hist = pn if n == 0 else torch.cat([p_hist, pn], dim=-1)
      y_hist = y.unsqueeze(-1) if n == 0 else torch.cat([y_hist, y.unsqueeze(-1)], dim=-1)

      if self.enable_bottleneck:
        z_hist = z.unsqueeze(-1) if n == 0 else torch.cat([z_hist, z.unsqueeze(-1)], dim=-1)
        m_hist = mus.unsqueeze(-1) if n == 0 else torch.cat([m_hist, mus.unsqueeze(-1)], dim=-1)
        s_hist = sigmas.unsqueeze(-1) if n == 0 else torch.cat([s_hist, sigmas.unsqueeze(-1)], dim=-1)

      h_hist.append(h)

      newly_halted = torch.logical_and(
        keep_processing, torch.logical_or(
          (torch.sum(p_hist, dim=-1) > 1 - eps_halting) * enable_halting, 
          torch.tensor(n + 1 == M)))
      
      halted_at[newly_halted] = n
      keep_processing[newly_halted] = False
      n += 1

    # --- evaluation: store remaining outputs
    yt[~eval_halted, :] = y[~eval_halted, :]
    nt[~eval_halted] = n
    
    # --- training: formulate cumuluative halting probabilities as valid probability distributions
    halt_range = torch.arange(p_hist.shape[1]).unsqueeze(-1).T.repeat(xi.shape[0], 1)
    halted_at_tiled = halted_at.unsqueeze(-1).repeat(1, halt_range.shape[1])
    halt_mask = halt_range <= halted_at_tiled
    ph = p_hist * halt_mask
    ph = ph / torch.sum(ph, dim=-1, keepdim=True)

    # ponder costs are computed differently for ponder net, so just return `pt` as all zero.
    pt = torch.zeros((xi.shape[0],))

    addtl = {'p_halt': ph, 'y': y_hist, 'halted_at': halted_at, 'halt_mask': halt_mask}

    if self.enable_bottleneck:
      addtl['zv'] = z_hist
      addtl['mv'] = m_hist
      addtl['sv'] = s_hist

    addtl['h'] = torch.stack(h_hist, dim=-1)

    # @TODO: sx should be updated to new hidden state if processing sequential data

    return sx, yt, pt, nt, addtl

  def act_step(self, xi: torch.Tensor, sx: torch.Tensor, eps_halting: float, M: int):
    """"""
    sn = sx
    pt = 0. # cumuluative halting probability
    p_hist = []
    y_hist = []
    s_hist = []
    c_hist = []
    keep_processing = torch.ones((xi.shape[0],), dtype=bool)
    n = 0
    while n < M:
      if not keep_processing.any(): break
      sn = self.rnn(xi, sn)
      h, c = self.extract_cell_states(sn)
      z, mus, sigmas = self.compute_z(h, c)
      if self.enable_bottleneck: h, c = self.compute_latent_to_hidden_state(z)
      y = self.hidden_state_to_output(h)
      pn = torch.sigmoid(self.halt(h)).squeeze(1)
      sn = self.update_cell_state(h, c, sn)
      pt += pn
      p_hist.append(pn)
      s_hist.append(h)
      if self.rnn_cell_type == 'lstm': c_hist.append(c)
      y_hist.append(y)
      keep_processing[pt >= 1. - eps_halting] = False
      n += 1

    p_hist = torch.stack(p_hist, dim=-1)
    y_hist = torch.stack(y_hist, dim=-1)
    s_hist = torch.stack(s_hist, dim=-1)
    if self.rnn_cell_type == 'lstm': c_hist = torch.stack(c_hist, dim=-1)

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

    addtl = {'z': z, 'mus': mus, 'sigmas': sigmas, 'h': s_hist, 'p_halt': ph}

    return sx, yt, pt, nt, addtl
  
  def forward(
    self, x: torch.Tensor, sx: torch.Tensor = None, eps_halting = 1e-2, M = 14,
    step_type: str = 'act', num_fixed_ticks: int = 6, min_lambda_p: float = None):
    """
    """
    variable_len_output_names = ['h', 'p_halt', 'y', 'halt_mask', 'zv', 'mv', 'sv']

    T = x.shape[-1]
    P = []
    Y = []
    N = []
    Addtl = {}

    for t in range(T):
      xi = x.select(-1, t)

      if step_type == 'act':
        sx, yt, pt, nt, addtl = self.act_step(xi, sx, eps_halting, M)
      elif step_type == 'fixed':
        sx, yt, pt, nt, addtl = self.fixed_ticks_step(sx, xi, num_fixed_ticks)
      elif step_type == 'ponder-net':
        assert T == 1, 'sequential input not yet correctly handled'
        sx, yt, pt, nt, addtl = self.ponder_net_step(xi, sx, eps_halting, M, min_lambda_p)
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
      if k not in variable_len_output_names and len(Addtl[k]) > 0: 
        Addtl[k] = torch.stack(Addtl[k], dim=-1)
    return Y, P, N, Addtl

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

  def act_step(
    self, sx: torch.Tensor, xi: torch.Tensor, eps_halting: float, M: int, 
    fixed_M: bool, non_mf_state: bool):
    """"""
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
    if non_mf_state:
      st = s_hist.select(-1, -1)
      if self.rnn_cell_type == 'lstm': ct = c_hist.select(-1, -1)
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
    assert step_type in ['act', 'fixed', 'fixed-act', 'non-mf-act', 'act-non-mf-state']
    act_step_types = ['act', 'fixed-act', 'act-non-mf-state']

    T = x.shape[-1]
    P = []
    Y = []
    N = []
    Addtl = {}

    for t in range(T):
      xi = x.select(-1, t)

      if step_type in act_step_types:
        fixed_M = step_type == 'fixed-act'
        use_M = num_fixed_ticks if fixed_M else M
        non_mf_state = step_type == 'act-non-mf-state'
        sx, yt, pt, nt, addtl = self.act_step(sx, xi, eps_halting, use_M, fixed_M, non_mf_state)
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

def dict_to_str(d: dict):
  v = {**d}
  for k in v:
    if isinstance(v[k], float): v[k] = np.round(v[k], 4)
  s = str(v)
  s = s.replace('{', '').replace('}', '').replace(', ', ' ')
  return s

def fix_dict_for_matfile(d: dict):
  d = {**d}
  for k in d:
    if d[k] is None: d[k] = -1.
  return d

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

def compute_entropy(x: torch.Tensor, dim: int) -> torch.Tensor:
  y = torch.log(x)
  y[~torch.isfinite(y)] = 0.
  ent = -torch.sum(x * y, dim=dim)
  return ent

def p_halt_entropy_comparison(p_halt: torch.Tensor):
  ent_diffs = 0.
  for i in range(len(p_halt)):
    nt_max = p_halt[i].shape[1]
    p_nt_max = 1. / nt_max
    un_entropy = -np.log(p_nt_max)
    lp = torch.log(p_halt[i].detach().cpu())
    lp[~torch.isfinite(lp)] = 0.
    ent = -torch.sum(lp * p_halt[i], dim=1)
    ent_diffs += torch.mean(ent).item() - un_entropy
  return ent_diffs / len(p_halt)

def logits_to_class_pred(y):
  nc = y.shape[1]
  if nc == 1:
    # binary cross entropy
    yh = torch.sigmoid(y).squeeze(1)
    return (yh > 0.5).type(torch.long)
  else:
    return torch.argmax(torch.softmax(y, dim=1), dim=1)

def train(
  info, enc: Union[RecurrentClassifier, VRNNClassifier], foward_fn, data_train: DataLoader, 
  loss_fn, epoch_cb, num_epochs: int, lr: float, random_seed: int, gradient_clip: float):
  """"""
  torch.manual_seed(random_seed)
  optim = torch.optim.Adam([*enc.parameters()], lr=lr)

  for e in range(num_epochs):
    for xs, ys, intermediates in data_train:
      y, p, n, addtl = foward_fn(enc, xs)
      err_rate = sequence_error_rate(ys, logits_to_class_pred(y))

      optim.zero_grad()
      L, components = loss_fn(y, ys, p, addtl)
      L.backward()
      if gradient_clip is not None: torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
      optim.step()

      nf = n.type(torch.float)
      mu_n = torch.mean(nf)
      max_n = torch.max(nf)

      print(
        f'{e+1} of {num_epochs} | Loss: {L.item():.3f}' + 
        f' | N: {mu_n.item():.2f} (max {max_n.item()}) | Acc: {(1. - err_rate.item()):.3f}' + 
        f' | {dict_to_str(components)} | ({info[0]+1} of {info[1]})', flush=True
      )
      
    epoch_cb(enc, e, num_epochs)

def do_evaluate(enc: Union[RecurrentClassifier, VRNNClassifier], foward_fn, data: DataLoader, loss_fn):
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

    ph_dist = None
    if 'p_halt' in addtl:
      ph = addtl['p_halt']
      assert len(ph) == 1, 'Sequences not yet handled'
      p_halt_rel_entropy = p_halt_entropy_comparison(ph)
      ph_dist = torch.mean(ph[0], dim=0).detach().cpu().numpy()
    else:
      p_halt_rel_entropy = float('nan')

    sigmas = None
    if 'sv' in addtl:
      sigmas = torch.mean(addtl['sv'][0], dim=[0, 1]).detach().cpu().numpy()
    mus = None
    if 'mv' in addtl:
      mus = torch.mean(addtl['mv'][0], dim=[0, 1]).detach().cpu().numpy()

    L, loss_components = loss_fn(y, ys, p, addtl)

    res = {
      'acc': seq_acc.item(), 'err_rate': err_rate.item(), 
      'ticks': n.float().mean().item(), 'p_halt_rel_entropy': p_halt_rel_entropy}
    
    if ph_dist is not None: res['p_halt_distribution'] = ph_dist
    if sigmas is not None: res['encoder_sigma'] = sigmas
    if mus is not None: res['encoder_mu'] = mus
    for src_key_name in loss_components:
      dst_key_name = f'{src_key_name}_loss'
      res[dst_key_name] = loss_components[src_key_name]

    return res, addtl
  
def logic_task_extract_op_indices(intermediates, last_only=True):
  num_ops = intermediates[:, :, -1, :][:, -1, -1]
  op_indices = intermediates[:, :, 0, :]
  if last_only:
    op_indices = op_indices[torch.arange(op_indices.shape[0]), num_ops-1, -1]
  else:
    op_indices = op_indices[torch.arange(op_indices.shape[0]), :, -1]
  return op_indices

def decode_internal_reprs_logic_task_alt(
  enc: Union[RecurrentClassifier, VRNNClassifier], foward_fn, loss_fn, batch_size: int):
  """"""
  # --- train classifier
  num_ops = 10
  restrict_clfs_within_valid_ticks = False

  def train_clfs():
    data_prep = prepare_logic_task(batch_size=batch_size, num_ops=num_ops, fixed_num_ops=num_ops, seq_len=1)
    _, addtl = do_evaluate(enc, foward_fn, data_prep, loss_fn)

    last_ticki = addtl['halted_at'][:, -1]  # last step of sequence
    hs = addtl['h'][-1]
    max_t = hs.shape[-1]

    op_indices = logic_task_extract_op_indices(addtl['intermediates'], last_only=False)

    num_ops_eval = num_ops
    # num_ops_eval = min(num_ops_eval, 4)
    # max_t = min(max_t, 4)

    clfs = []
    for oi in range(num_ops_eval):
      print(f'{oi+1} of {num_ops_eval}')
      for ti in range(max_t):
        xs = hs[:, :, ti].detach().cpu().numpy()
        ys = op_indices[:, oi].detach().cpu().numpy()
        if restrict_clfs_within_valid_ticks:
          # don't consider hidden states beyond the final chosen tick
          m = ti <= last_ticki.detach().cpu().numpy()
          xs = xs[m, :]
          ys = ys[m]
        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, loss='log_loss'))
        clf.fit(xs, ys)
        clfs.append({'oi': oi, 'ti': ti, 'clf': clf})
    return clfs
  
  clfs = train_clfs()

  data_prep = prepare_logic_task(
  batch_size=1_000, num_ops=num_ops, fixed_num_ops=num_ops, seq_len=1)
  _, addtl = do_evaluate(enc, foward_fn, data_prep, loss_fn)
  last_ticki = addtl['halted_at'][:, -1]  # last step of sequence
  hs = addtl['h'][-1]
  max_t = hs.shape[-1]
  op_indices = logic_task_extract_op_indices(addtl['intermediates'], last_only=False)

  ps = np.zeros((max_t, num_ops))
  for ti in range(max_t):
    m = ti <= last_ticki
    if not restrict_clfs_within_valid_ticks: m[:] = True
    xs = hs[m, :, ti]
    for oi in range(num_ops):
      ys = op_indices[m, oi]
      rel_clf = [*filter(lambda x: x['ti'] == ti and x['oi'] == oi, clfs)]
      assert len(rel_clf) == 1, 'No classifier matched'      
      rel_clf = rel_clf[0]['clf']
      ps[ti, oi] = rel_clf.score(xs.detach().cpu().numpy(), ys.detach().cpu().numpy())

  rd = {'ps': ps}
  return rd
  
def decode_internal_reprs_logic_task(
  enc: Union[RecurrentClassifier, VRNNClassifier], foward_fn, loss_fn, batch_size: int):
  """
  """
  num_ops = 10
  data_prep = prepare_logic_task(batch_size=batch_size, num_ops=num_ops, fixed_num_ops=None, seq_len=1)
  _, addtl = do_evaluate(enc, foward_fn, data_prep, loss_fn)
  last_ticki = addtl['halted_at'][:, -1]  # last step of sequence
  last_h = addtl['h'][-1]
  last_h = last_h[torch.arange(last_h.shape[0]), :, last_ticki]
  op_indices = logic_task_extract_op_indices(addtl['intermediates'])

  clfs = []
  clf_ys = [addtl['ys'][:, -1].detach().cpu().numpy(), op_indices.detach().cpu().numpy()]
  clf_ncs = [enc.nc, num_ops]
  for clf_Y in clf_ys:
    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, loss='log_loss'))
    clf_X = last_h.detach().cpu().numpy()
    clf.fit(clf_X, clf_Y)
    clfs.append(clf)

  # now evaluate over ticks, using sequence of operations that does not have duplicates
  data_prep = prepare_logic_task(
    batch_size=1_000, num_ops=10, fixed_num_ops=None, seq_len=1, sample_ops_with_replacement=False)
  _, addtl = do_evaluate(enc, foward_fn, data_prep, loss_fn)
  hs = addtl['h']
  nt = max([x.shape[-1] for x in hs])
  # @TODO:
  op_indices_test = logic_task_extract_op_indices(addtl['intermediates']).detach().cpu().numpy()

  class_pred_ys = []
  class_prob_sets = []
  for clfi, clf in enumerate(clfs):
    clf_obj = clf.named_steps['sgdclassifier']
    # clf_coef = clf_obj.coef_
    class_preds = []
    class_probs = []
    for si in range(len(hs)):
      hi = hs[si]
      class_pred = torch.ones(hi.shape[0], nt) * -1
      class_prob = torch.zeros(hi.shape[0], max(clf_ncs), nt)
      for pi in range(hi.shape[2]):
        hp = hi[:, :, pi].detach().cpu().numpy()
        yp = clf.predict(hp)
        sc = clf.predict_proba(hp)
        class_pred[:, pi] = torch.tensor(yp)
        class_prob[:, :clf_ncs[clfi], pi] = torch.tensor(sc)
      class_preds.append(class_pred.detach().cpu().numpy())
      class_probs.append(class_prob.detach().cpu().numpy())
    class_pred_ys.append(np.stack(class_preds, axis=-1))
    class_prob_sets.append(np.stack(class_probs, axis=-1))
  
  class_pred_ys = np.stack(class_pred_ys, axis=-1)
  class_prob_sets = np.stack(class_prob_sets, axis=-1)

  ints = addtl['intermediates']
  rd = {
    'n': addtl['n'].detach().cpu().numpy(), 
    'pred_ys': class_pred_ys, 'intermediates': ints.detach().cpu().numpy(),
    'class_probs': class_prob_sets
  }
  return rd

def evaluate_generalization(
  ctx: EvaluateContext, enc: RecurrentClassifier, forward_fn, loss_fn, train_epoch: int, hps: dict):
  # evaluate generalization performance when sequences are longer than those seen during training
  if hps['task_type'] == 'logic':
    for i in range(3):
      slen = 4 + i
      hp_res = {**hps}
      hp_res['seq_len'] = slen
      data_prep = prepare_logic_task(
        batch_size=ctx.default_batch_size, num_ops=10, fixed_num_ops=None, seq_len=slen)
      evaluate_baseline(ctx, enc, forward_fn, data_prep, loss_fn, train_epoch, hp_res)

def evaluate_hidden_representations(
  ctx: EvaluateContext, enc: RecurrentClassifier, forward_fn, loss_fn, train_epoch: int, hps: dict):
  # decode hidden state representation(s) of output
  if hps['task_type'] == 'logic':
    hp_res = {**hps}
    # decode_res = decode_internal_reprs_logic_task(enc, forward_fn, loss_fn, ctx.default_batch_size)
    decode_res = decode_internal_reprs_logic_task_alt(enc, forward_fn, loss_fn, ctx.default_batch_size)
    decode_res['hps'] = fix_dict_for_matfile(hp_res)
    if ctx.do_save:
      matname = GenCPFnameFn(hp_res, True)(train_epoch).replace('.pth', '.mat')
      savemat(os.path.join(ctx.save_p, 'decoding-alt', matname), decode_res)

def evaluate_varying_difficulty(
  ctx: EvaluateContext, enc: RecurrentClassifier, forward_fn, loss_fn, train_epoch: int, hps: dict):
  # evaluate performance when varying example difficulty
  if hps['task_type'] == 'logic':
    num_ops = [*np.arange(0, 10, 2)] + [9]
    for it, i in enumerate(num_ops):
      print(f'\t{it+1} of {len(num_ops)}')
      data_prep = prepare_logic_task(batch_size=ctx.default_batch_size, num_ops=10, fixed_num_ops=i+1, seq_len=1)
      hps_fixed = {**hps}
      hps_fixed['fixed_num_ops'] = i + 1
      evaluate_baseline(ctx, enc, forward_fn, data_prep, loss_fn, train_epoch, hps_fixed)

def evaluate_fixed_ticks(
  ctx: EvaluateContext, enc: RecurrentClassifier, forward_fn, data: DataLoader, 
  loss_fn, train_epoch: int, hps: dict):
  """"""
  for t in [*np.arange(0, 16, 2)]:
    num_fixed_ticks = t + 1
    for st in ['fixed', 'fixed-act']:
      forward_fixed_fn = lambda enc, xs: enc(xs, step_type=st, num_fixed_ticks=num_fixed_ticks)
      res, addtl = do_evaluate(enc, forward_fixed_fn, data, loss_fn)
      hp_res = {**hps}
      hp_res['step_type'] = st
      hp_res['num_fixed_ticks'] = num_fixed_ticks
      save_fname = GenCPFnameFn(hp_res, True)(train_epoch).replace('.pth', '.mat')
      res['hps'] = fix_dict_for_matfile(hp_res)
      if ctx.do_save:
        savemat(os.path.join(ctx.save_p, save_fname.replace('.pth', '.mat')), res)
  
def evaluate(
  ctx: EvaluateContext, enc: RecurrentClassifier, forward_fn, data: DataLoader, 
  loss_fn, train_epoch: int, hps: dict):
  """"""
  hps = {**hps}
  hps['num_fixed_ticks'] = -1
  hps['seq_len'] = 3

  # if True: evaluate_baseline(ctx, enc, forward_fn, data, loss_fn, train_epoch, {**hps})
  if True: evaluate_varying_difficulty(ctx, enc, forward_fn, loss_fn, train_epoch, hps)
  # if True: evaluate_hidden_representations(ctx, enc, forward_fn, loss_fn, train_epoch, hps)
  # if False: evaluate_generalization(ctx, enc, forward_fn, loss_fn, train_epoch, hps)
  # if False: evaluate_fixed_ticks(ctx, enc, forward_fn, data, loss_fn, train_epoch, {**hps})

def evaluate_baseline(
  ctx: EvaluateContext, enc: RecurrentClassifier, forward_fn, data: DataLoader, 
  loss_fn, train_epoch: int, hps: dict):
  """"""
  hp_res = {**hps}
  hp_res['num_fixed_ticks'] = -1
  res, addtl = do_evaluate(enc, forward_fn, data, loss_fn)
  res['hps'] = fix_dict_for_matfile(hp_res)
  if ctx.do_save:
    savemat(os.path.join(ctx.save_p, GenCPFnameFn(hp_res, True)(train_epoch).replace('.pth', '.mat')), res)

# ------------------------------------------------------------------------------------------------

def prepare_logic_task(
    *, batch_size: int, num_ops: int, fixed_num_ops: int, 
    seq_len: int = 3, fixed_num_ops_p = None, **kwargs):
  """"""
  ds = LogicDataset(
    batch_size=batch_size, seq_len=seq_len, num_ops=num_ops, 
    num_samples=None, fixed_num_ops=fixed_num_ops, fixed_num_ops_p=fixed_num_ops_p, **kwargs)
  return DataLoader(ds, batch_size=batch_size)

def prepare_parity_task(*, batch_size: int, vector_length: int):
  ds = ParityDataset(batch_size=batch_size, vector_length=vector_length)
  return DataLoader(ds, batch_size=batch_size)

def prepare_data(task_type: str, batch_size: int, task_params: dict):
  if task_type == 'logic':
    nc = 2
    num_ops = 10
    input_dim = num_ops * 10 + 2
    stride = 1
    data_train = prepare_logic_task(
      batch_size=batch_size, num_ops=num_ops, fixed_num_ops=None, **task_params)

  elif task_type == 'addition':
    max_num_digits = 5
    input_dim = max_num_digits * 10
    stride = 11
    nc = stride * (max_num_digits + 1)
    ldp = {'batch_size': batch_size, 'seq_len': 5, 'max_num_digits': max_num_digits}
    data_train = DataLoader(AdditionDataset(**ldp), batch_size=batch_size)

  elif task_type == 'parity':
    assert 'vector_length' in task_params
    input_dim = task_params['vector_length']
    nc = 2
    data_train = prepare_parity_task(batch_size=batch_size, **task_params)

  else: assert False

  return data_train, input_dim, nc

def discrete_uniform_dist(n: int):
  return torch.ones((n,)) / n

def geometric_dist(n: int, p: float):
  k = (1 + np.arange(n)).astype(np.float64)
  geo = np.power((1. - p), k) * p
  geo = geo / np.sum(geo)
  return geo

def discrete_kl_divergence(p, q, dim, eps: float = 1e-12):
  p = p + eps
  q = q + eps

  p = p / torch.sum(p, dim=dim, keepdim=True)
  q = q / torch.sum(q, dim=dim, keepdim=True)

  lp = torch.log(p)
  lq = torch.log(q)

  plp = p * lp
  plp[p == 0] = 0.

  plq = p * lq
  plq[(q == 0) & (p == 0)] = 0.
  plq[(q == 0) & (p > 0)] = float('inf')

  d = torch.sum(plp - plq, dim)
  return d

def kl_full_gaussian(mu: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
  k = mu.size(-1)
  # trace(Σ) = ‖L‖_F² because trace(L Lᵀ) = Σᵢⱼ L_{ij}²
  trace_term = (L ** 2).sum(dim=(-2, -1))
  # log|Σ| = 2·Σ_i log L_ii (diagonal of a Cholesky factor is > 0)
  log_det = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
  quad_term = mu.pow(2).sum(-1) # μᵀμ
  kl = 0.5 * (trace_term + quad_term - k - log_det)
  return kl.sum()

def kl_divs_diag_gaussian(mus: torch.Tensor, sigs: torch.Tensor):
  vars = sigs.pow(2)
  return -0.5 * (1.0 + vars.log() - mus.pow(2) - vars)

def kl_div(mus: torch.Tensor, sigs: torch.Tensor):
  for i in range(mus.shape[-1]):
    mu, sig = mus.select(-1, i), sigs.select(-1, i)
    kl = kl_full_gaussian(mu, sig)
    s = kl if i == 0 else s + kl
  return s / mus.shape[-1]

class TrainCBFn(object):
  def __init__(self, *, do_save: bool, cp_save_interval: int, gen_cp_fname_fn, hps: dict, root_p: str):
    self.do_save = do_save
    self.cp_save_interval = cp_save_interval
    self.gen_cp_fname_fn = gen_cp_fname_fn
    self.hps = hps
    self.root_p = root_p
    self.last_cp_only = cp_save_interval is None

  def get_save_p(self, e: int):
    # cp_fname = self.gen_cp_fname_fn(e)
    cp_fname = f'{random_str(8)}.pth'
    return os.path.join(self.root_p, 'data', cp_fname)

  def __call__(self, enc, e: int, num_epochs: int):
    if not self.do_save: return
    is_last = (e + 1) == num_epochs
    do_save = is_last or ((e % self.cp_save_interval == 0) and not self.last_cp_only)
    if not do_save: return
    cp = {'state': enc.state_dict(), 'hps': self.hps}
    torch.save(cp, self.get_save_p(e))

class GravesGenCPFnameFn(object):
  def __init__(self, hps: dict):
    self.hps = {k: hps[k] for k in ['ponder_cost', 'step_type', 'task_type']}

  def __call__(self, e: int):
    res = f'act-checkpoint-{make_cp_id(self.hps)}-{e}.pth'
    return res

class GenCPFnameFn(object):
  def __init__(self, hps: dict, eval_mode: bool):
    self.hps = hps
    self.eval_mode = eval_mode
    self.excl_hps = ['loss_fn_type', 'eps_halting', 'ponder_penalty_type', 'explore_weight']
    self.excl_hps_if_none = ['min_lp']

  def include_hp(self, k: str) -> bool:
    if k in self.excl_hps: return False
    if k in self.excl_hps_if_none and self.hps[k] is None: return False
    return True

  def __call__(self, e: int):
    hp_gen = {k: self.hps[k] for k in self.hps if self.include_hp(k)}
    res = f'act-checkpoint-{make_cp_id(hp_gen)}-{e}.pth'
    if self.eval_mode:
      res = res[:min(len(res), 128)].replace('.pth', '')
      res += random_str(8)
      res = f'{res}.pth'
    return res
  
class VRNNForwardFn(object):
  def __init__(
    self, *, M: int, num_fixed_ticks: int, step_type: str, eps_halting: float, min_lambda_p: float):
    """"""
    self.M = M
    self.num_fixed_ticks = num_fixed_ticks
    self.step_type = step_type
    self.eps_halting = eps_halting
    self.min_lambda_p = min_lambda_p

  def __call__(self, enc: VRNNClassifier, xs: torch.Tensor):
    return enc(
      xs, num_fixed_ticks=self.num_fixed_ticks, M=self.M,
      step_type=self.step_type, eps_halting=self.eps_halting, min_lambda_p=self.min_lambda_p)

class ForwardFn(object):
  def __init__(self, *, step_type: str, num_fixed_ticks: int, M: int):
    self.step_type = step_type
    self.num_fixed_ticks = num_fixed_ticks
    self.M = M

  def __call__(self, enc: RecurrentClassifier, xs: torch.Tensor): 
    return enc(xs, step_type=self.step_type, num_fixed_ticks=self.num_fixed_ticks, M=self.M)

class GravesLossFn(object):
  def __init__(self, *, ponder_cost: float):
    self.ponder_cost = ponder_cost
    self.reduction = 'sum'
    # self.reduction = 'mean'

  def __call__(self, y, ys, p, addtl: dict):
    L_acc = nn.functional.cross_entropy(y, ys, reduction=self.reduction)
    if self.reduction == 'sum':
      L_ponder = torch.sum(p)
    else:
      assert self.reduction == 'mean'
      L_ponder = torch.mean(p)
    res = L_acc + L_ponder * self.ponder_cost
    return res, {}
  
class PonderNetLossFn(object):
  def __init__(
      self, *, 
      lambda_p: float = 0.2,
      ponder_weight: float = 1. * 1e-2,
      accuracy_weight: float = 1.,
      complexity_weight: float = 1. * 1e-2,
      explore_weight: float = 0.,
      ponder_penalty_type: str = 'kl-geometric',
      weight_normalization_type: str = 'none'):
    """"""
    assert ponder_penalty_type in ['kl-geometric', 'expected-ticks']
    assert weight_normalization_type in ['none', 'divide_sum']

    self.disable_ponder_weight = False
    self.last_tick_only = False
    self.ponder_penalty_type = ponder_penalty_type
    self.weight_normalization_type = weight_normalization_type

    self.lambda_p = torch.tensor(lambda_p)
    self.accuracy_weight = accuracy_weight
    self.ponder_weight = ponder_weight * (1. - self.disable_ponder_weight)
    self.complexity_weight = complexity_weight
    self.explore_weight = explore_weight
    self.eps = 1e-10
    self.step = 0

  def __call__(self, y, ys, p, addtl: dict):
    T = ys.shape[1] # sequence length

    L_ce = torch.tensor(0.)
    L_ponder_prior = torch.tensor(0.)
    L_complexity = torch.tensor(0.)
    L_explore = torch.tensor(0.)

    for t in range(T):
      yi = ys[:, t]
      halted_at = addtl['halted_at'][:, t]
      # non_halted = addtl['halt_mask'][t]
      p_halt = addtl['p_halt'][t]
      y_hat = addtl['y'][t]

      # --- accuracy term:
      nc = y_hat.shape[1]
      if nc == 1:
        # binary cross entropy
        p_hat = torch.min(torch.tensor(1. - self.eps), torch.sigmoid(y_hat) + self.eps).squeeze(1)
        yi_t = yi.repeat(p_hat.shape[1], 1).T.type(torch.float32)
        ce = -(yi_t * torch.log(p_hat) + (1. - yi_t) * torch.log(1. - p_hat))

      else:
        # cross entropy
        p_hat = torch.softmax(y_hat, dim=1)
        # y_pred = torch.argmax(torch.softmax(y_hat, dim=1), dim=1)
        ce = -torch.log(torch.min(
          torch.tensor(1. - self.eps), 
          p_hat[torch.arange(p_hat.shape[0]), yi, :] + self.eps))

      if self.last_tick_only:
        ce = torch.select(ce, -1, -1)
        ce = ce.reshape((*ce.shape, 1))
      else:
        ce = ce * p_halt

      L_ce += torch.mean(torch.sum(ce, dim=1))

      # --- ponder prior term:
      num_halt = halted_at + 1
      max_halt = torch.max(num_halt)
      geo_prior_k = torch.arange(max_halt).unsqueeze(-1).T.repeat(yi.shape[0], 1).type(torch.float32)

      if self.ponder_penalty_type == 'kl-geometric':
        # KL-div between halting probabilities and geometric distribution
        geo_prior = torch.pow((1 - self.lambda_p), geo_prior_k) * self.lambda_p
        geo_prior = geo_prior / torch.sum(geo_prior, dim=1, keepdim=True)
        ponder_prior_kl_term = discrete_kl_divergence(p_halt, geo_prior, 1)
        L_ponder_prior += torch.mean(ponder_prior_kl_term)

      elif self.ponder_penalty_type == 'expected-ticks':
        # expected # ticks
        expected_ticks = torch.sum(p_halt * (1. + geo_prior_k), dim=1)
        L_ponder_prior += torch.mean(expected_ticks)

      else: assert False

      # --- exploitation penalty (KL from uniform):
      un_dist = discrete_uniform_dist(p_halt.shape[1]).unsqueeze(-1).T.repeat(p_halt.shape[0], 1)
      uniform_kl = discrete_kl_divergence(p_halt, un_dist, 1)
      L_explore += torch.mean(uniform_kl)

      # --- complexity term: 
      if 'mv' in addtl:
        mus = addtl['mv'][t]
        sigs = addtl['sv'][t]
        # dim=1 -> mean across latent dimensions, preserving batch and sequence
        kl_iso_gauss = torch.mean(kl_divs_diag_gaussian(mus, sigs), dim=1)
        # expectation across time steps
        kl_iso_gauss *= p_halt
        L_complexity += torch.mean(torch.sum(kl_iso_gauss, dim=1))

    L_ce /= T
    L_ponder_prior /= T
    L_complexity /= T
    L_explore /= T

    aw, pw, cw, ew = self.accuracy_weight, self.ponder_weight, self.complexity_weight, self.explore_weight
    # total loss
    L = L_ce*aw + L_ponder_prior*pw + L_complexity*cw + L_explore*ew

    if self.weight_normalization_type == 'divide_sum':
      weight_sum = aw + pw + cw + ew
      L /= weight_sum
    else:
      assert self.weight_normalization_type == 'none'

    components = {
      'accuracy': L_ce.item(), 'ponder_prior': L_ponder_prior.item(), 
      'complexity': L_complexity.item(), 'explore': L_explore.item()
    }

    return L, components

class VariationalLossFn(object):
  def __init__(self, *, ponder_cost: float, beta: float, weight_normalization_type: str):
    assert weight_normalization_type in ['norm', 'none', 'sums_to_1']
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
    elif self.weight_normalization_type == 'sums_to_1':
      weights /= np.sum(weights)
    else: assert False
    # return L_acc + ponder_cost*L_ponder + beta*L_kl
    res = weights[0]*L_acc + weights[1]*L_ponder + weights[2]*L_kl
    return res, {}
  
def prepare_eval_from_checkpoint(cp: dict, data_train, input_dim: int, nc: int):
  hps = cp['hps']
  state = cp['state']
  # @TODO: Only supporting basic hyperparameter combinations
  assert hps['model_type'] == 'vrnn'
  assert hps['loss_fn_type'] == 'ponder-net'

  enc = VRNNClassifier(
    input_dim=input_dim, hidden_dim=hps['rnn_hidden_dim'],
    rnn_cell_type=hps['rnn_cell_type'], nc=nc, bottleneck_K=hps['bottleneck_K']
  )
  forward_fn = VRNNForwardFn(
    M=hps['M'], num_fixed_ticks=hps['num_fixed_ticks'], 
    step_type=hps['step_type'], eps_halting=hps['eps_halting'],
    min_lambda_p=hps['min_lp'])
  loss_fn = PonderNetLossFn(
    complexity_weight=hps['beta'], ponder_weight=hps['ponder_cost'], lambda_p=hps['lp'], 
    ponder_penalty_type=hps['ponder_penalty_type'], explore_weight=hps['explore_weight'],
    weight_normalization_type=hps['weight_normalization_type'], accuracy_weight=hps['accuracy_weight'])
  
  enc.eval()
  enc.load_state_dict(state)
  return enc, forward_fn, loss_fn

def prepare(
  *, ponder_cost: float, do_save: bool, step_type: str, num_fixed_ticks: int, hps: dict,
  data_train: DataLoader, input_dim: int, rnn_hidden_dim: int, 
  nc: int, eval_epoch: int, bottleneck_K: int, beta: float, 
  M: int, weight_normalization_type: str, root_p: str, loss_fn_type: str, 
  model_type: str, rnn_cell_type: str, eps_halting: float, lambda_p: float, min_lambda_p: float,
  ponder_penalty_type: str, explore_weight: float, accuracy_weight: float, cp_save_interval: int):
  """"""

  assert model_type in ['rvib', 'vrnn']
  assert loss_fn_type in ['graves', 'variational', 'ponder-net']

  rnn_hidden_dim = hps['rnn_hidden_dim'] = rnn_hidden_dim
  rnn_cell_type = hps['rnn_cell_type'] = rnn_cell_type

  hps['ponder_cost'] = ponder_cost
  hps['num_fixed_ticks'] = num_fixed_ticks
  hps['step_type'] = step_type
  hps['model_type'] = model_type
  hps['loss_fn_type'] = loss_fn_type
  hps['eps_halting'] = eps_halting
  hps['lp'] = lambda_p
  hps['min_lp'] = min_lambda_p
  hps['ponder_penalty_type'] = ponder_penalty_type
  hps['explore_weight'] = explore_weight
  hps['accuracy_weight'] = accuracy_weight

  if model_type == 'vrnn':
    enc = VRNNClassifier(
      input_dim=input_dim, hidden_dim=rnn_hidden_dim,
      rnn_cell_type=rnn_cell_type, nc=nc, bottleneck_K=bottleneck_K
    )
    forward_fn = VRNNForwardFn(
      M=M, num_fixed_ticks=num_fixed_ticks, step_type=step_type, eps_halting=eps_halting,
      min_lambda_p=min_lambda_p)
  elif model_type == 'rvib':
    enc = RecurrentClassifier(
      input_dim=input_dim, hidden_dim=rnn_hidden_dim,
      rnn_cell_type=rnn_cell_type, nc=nc, bottleneck_K=bottleneck_K
    )
    forward_fn = ForwardFn(step_type=step_type, num_fixed_ticks=num_fixed_ticks, M=M)
  else: assert False
  
  if loss_fn_type == 'graves': 
    gen_cp_fname_fn = GravesGenCPFnameFn(hps)
  else:
    gen_cp_fname_fn = GenCPFnameFn(hps, False)

  if eval_epoch is not None:
    enc.eval()
    cp = torch.load(os.path.join(root_p, 'data', gen_cp_fname_fn(eval_epoch)))
    enc.load_state_dict(cp['state'])

  train_cb_fn = TrainCBFn(
    do_save=do_save, cp_save_interval=cp_save_interval, gen_cp_fname_fn=gen_cp_fname_fn, 
    hps=hps, root_p=root_p)

  if loss_fn_type == 'graves':
    loss_fn = GravesLossFn(ponder_cost=ponder_cost)
  elif loss_fn_type == 'variational':
    loss_fn = VariationalLossFn(
      ponder_cost=ponder_cost, beta=beta, weight_normalization_type=weight_normalization_type
    )
  elif loss_fn_type == 'ponder-net':
    loss_fn = PonderNetLossFn(
      complexity_weight=beta, ponder_weight=ponder_cost, lambda_p=lambda_p, 
      ponder_penalty_type=ponder_penalty_type, explore_weight=explore_weight,
      weight_normalization_type=weight_normalization_type, accuracy_weight=accuracy_weight)
  else: assert False

  args = (enc, forward_fn, data_train, loss_fn, train_cb_fn, gen_cp_fname_fn)
  return args

# ------------------------------------------------------------------------------------------------

def random_str(n: int) -> str:
  return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))

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
  for i, args in enumerate(arg_sets): 
    print(f'{i+1} of {len(arg_sets)}')
    evaluate(*args)

def run(set_fn, arg_sets, num_processes: int):
  if num_processes <= 0:
    set_fn(arg_sets)
  else:
    pi = split_array_indices(len(arg_sets), num_processes)
    process_args = [[arg_sets[x] for x in y] for y in pi]
    processes = [Process(target=set_fn, args=(args,)) for args in process_args]
    for p in processes: p.start()
    for p in processes: p.join()

def find_files_in_directory(d: str):
  import glob
  return glob.glob(os.path.join(d, '*'))

def filter_files_by_date(file_paths: List[str], date: datetime, op: str) -> List[str]:
  """"""
  assert op in ['gt', 'lt', 'ge', 'le']
  ops = {'gt': lambda x, y: x > y, 'lt': lambda x, y: x < y,
         'ge': lambda x, y: x >= y, 'le': lambda x, y: x <= y}

  filtered_files = []
  cutoff_timestamp = date.timestamp()
  for path in file_paths:
    try:
      if os.path.isfile(path):
        mtime = os.path.getmtime(path)
        if ops[op](mtime, cutoff_timestamp):
          filtered_files.append(path)
    except OSError:
      # Skip files that can't be accessed
      continue  
  return filtered_files

def prepare_eval_from_directory(
  cps: List[str], eval_ctx: EvaluateContext, task_type: str, batch_size: int, task_params: dict):
  """"""
  
  arg_sets = []
  for cp in cps:
    data_train, input_dim, nc = prepare_data(task_type, batch_size, task_params)

    res = torch.load(cp)
    enc, forward_fn, loss_fn = prepare_eval_from_checkpoint(res, data_train, input_dim, nc)

    tstamp = str(datetime.datetime.fromtimestamp(os.path.getmtime(cp)))
    hps = {**res['hps']}
    hps['epoch'] = -1
    hps['timestamp'] = tstamp
    hps['checkpoint_filename'] = os.path.split(cp)[1]

    arg_sets.append((eval_ctx, enc, forward_fn, data_train, loss_fn, -1, hps))

  return arg_sets

def main():
  # --- program hps
  # root_p = '/Users/nick/source/mattarlab/explore-variational-rnn/output/ponder-net-manipulate-network-size-mult-cps-large-n'
  root_p = '/Users/nick/source/mattarlab/explore-variational-rnn/output/ponder-net-simplex'
  do_save = True
  skip_existing = True
  dry_run = False
  evaluate_directory = True
  
  num_processes = 5
  replicate_graves = False
  replicate_ponder_net = True
  do_train = False
  if not do_train: num_processes = 0
  task_type = 'logic'
  task_params = {}
  task_params['fixed_num_ops_p'] = np.flip(geometric_dist(10, 0.3))
  # --- program hps

  # --- network / training hps
  match_rnn_hidden_dim_to_bottleneck_K = False
  ponder_penalty_type = 'kl-geometric'
  lambda_p = 0.
  explore_weight = 0.
  min_lambda_p = None
  eps_halting = 1e-2
  grad_clip = None  # before 6/24/25
  # grad_clip = 1.0 # on/after 6/24/25
  loss_fn_type = 'variational'
  model_type = 'vrnn'
  # model_type = 'rvib'
  # rnn_cell_type = 'lstm'
  rnn_cell_type = 'gru'
  # rnn_cell_type = 'rnn'
  rnn_hidden_dim = 512

  # rnn_cell_type = 'rnn'
  # rnn_hidden_dim = 2048

  cp_save_interval = 5_000
  train_batch_size = 32
  # eval_batch_size = 10_000
  eval_batch_size = 5_000
  num_train_epochs = 130_001
  step_type = 'act'
  # step_type = 'fixed'
  # step_type = 'act-non-mf-state'
  num_fixed_ticks = 6 # @NOTE: This isn't used unless step_type == 'fixed'
  lr = 1e-4
  M = 6
  M = 14
  weight_normalization_type = 'none'
  # weight_normalization_type = 'sums_to_1'
  ponder_costs = [1e-3, 1e-3 * 1.5, 1e-3 * 1.75, 1e-3 * 2]
  ponder_costs = [4e-3]
  # ponder_costs = [1e-3, 16e-3]
  bottleneck_Ks = [512]
  betas = [0, 1e-2, 1e-1, 2e-2, 3e-2] + [1e-2*(1/2), 1e-2*(1/3)]
  # betas = [0., 1e-2, 2e-2]
  betas = [1e-3, 1e-2, 1e-3 * 0.5, 1e-2 * 0.5, 1e-2 * 2]
  betas += [0.]
  betas = [1e-3]
  accuracy_weights = [1.]
  # --- network / training hps

  # betas = [0.]

  seeds = [61, 62, 63, 64, 65]
  # seeds = [seeds[0]]
  # seeds = [62, 63]
  seeds = seeds[:3]

  if not do_train:
    eval_epochs = list(np.arange(0, num_train_epochs, 2_000))
    # --- several epochs
    # eval_epochs = eval_epochs[::2]

    # --- just one epoch
    eval_epochs = [ eval_epochs[-1] ]
    # eval_epochs = [ 104000 ]

  else:
    # --- train, instead of eval
    eval_epochs = [None]

  # --- override params for Graves replication
  if replicate_graves:
    bottleneck_Ks = [None]
    betas = [0.]
    train_batch_size = 16
    ponder_costs = [1e-3, 1e-3*2, 1e-2]
    seeds = [61]
    lr = 1e-4
    M = 6
    weight_normalization_type = 'none'
    model_type = 'rvib'
    loss_fn_type = 'graves'
    rnn_cell_type = 'lstm'
    rnn_hidden_dim = 128
  # --- override params for replication

  # --- override params for PonderNet replication
  if replicate_ponder_net:
    lambda_p = 0.2
    bottleneck_Ks = [None]
    ponder_costs = [1e-2]
    train_batch_size = 128
    model_type = 'vrnn'
    loss_fn_type = 'ponder-net'
    rnn_cell_type = 'rnn'
    rnn_hidden_dim = 128
    lr = 3e-4
    M = 20
    task_type = 'logic'
    step_type = 'ponder-net'
    eps_halting = 0.05
    # num_train_epochs = 250_000
    num_train_epochs = 60_000
    # num_train_epochs = 100_000
    if not do_train: eval_epochs = [num_train_epochs - 1]

    # seeds = seeds[:3]
    # seeds = seeds[3:]

    if task_type == 'parity':
      task_params['vector_length'] = 64

    elif task_type == 'logic':
      # seeds = [*range(66, 71)]
      # seeds = [64, 65]

      rnn_hidden_dim = 256
      rnn_cell_type = 'gru'
      task_params['seq_len'] = 1
      # make harder examples moderately more likely
      task_params['fixed_num_ops_p'] = np.flip(geometric_dist(10, 0.3))
      bottleneck_Ks = [rnn_hidden_dim]
      # bottleneck_Ks = [64, 128, 256]
      # bottleneck_Ks = [512]
      betas = [0., 1e-3, 2e-3, 5e-3, 1e-2, 5e-2]
      # betas = [0.]
      lambda_p = 0.2
      # ponder_costs = [1e-2 * 2., 1e-2 * 4.]
      # ponder_costs = [1e-2*1, 1e-2*2, 1e-2*3, 1e-2*4, 1e-1]
      ponder_costs = [1e-2*1, 1e-2*3, 1e-1]
      # ponder_costs = [0., 5e-1]
      ponder_costs = [1.]
      betas = betas[:4]

      betas = [1e-4]
      ponder_costs = [1e-2, 3e-2, 1e-1, 5e-1, 1.]

      match_rnn_hidden_dim_to_bottleneck_K = True

      # --- simplex
      weight_normalization_type = 'divide_sum'
      accuracy_weights = sorted([1., 0.95, 0.8, 0.5, 1e-2, 1e-3])
      # --- end

      # --- single tick
      if False:
        min_lambda_p = 1. # force only a single tick
        ponder_costs = [0.]
        betas = [0.]
        bottleneck_Ks = [256, 512, 1024]
        seeds = seeds[:3]
      # --- end single tick

      # ponder_penalty_type = 'expected-ticks'
      # explore_weight = 1.
      # betas = [0.]
      # ponder_costs = [1e-4]
      # seeds = seeds[:1]
      # assert False

  # --- override params for replication
  if not do_train:
    eval_epochs = [*np.arange(0, num_train_epochs, cp_save_interval), num_train_epochs-1]

    if evaluate_directory:
      eval_d = os.path.join(root_p, 'data')
      cp_fs = find_files_in_directory(eval_d)
      if True:
        # cutoff = datetime.datetime(2025, 9, 2)
        cutoff = datetime.datetime(2025, 9, 8)
        cp_fs = filter_files_by_date(cp_fs, cutoff, 'lt')
      if False: # @TODO
        cp_fs = filter_files_by_epoch(cp_fs, num_train_epochs-1)
      eval_ctx = EvaluateContext(
        save_p=os.path.join(root_p, 'results'), do_save=do_save, default_batch_size=eval_batch_size)
      arg_sets = prepare_eval_from_directory(cp_fs, eval_ctx, task_type, eval_batch_size, task_params)
      run(eval_set, arg_sets, num_processes)
      return

  its = [*product(seeds, eval_epochs, ponder_costs, betas, accuracy_weights, bottleneck_Ks)]
  arg_sets = []

  for i, it in enumerate(its):
    print(f'{i+1} of {len(its)}')

    seed, eval_epoch, ponder_cost, beta, accuracy_weight, bottleneck_K = it
    hps = {}

    hps['task_type'] = task_type
    train_batch_size = hps['batch_size'] = train_batch_size
    seed = hps['seed'] = seed
    hps['bottleneck_K'] = bottleneck_K
    hps['beta'] = beta
    hps['M'] = M
    hps['weight_normalization_type'] = weight_normalization_type

    batch_size = train_batch_size if eval_epoch is None else eval_batch_size
    data_train, input_dim, nc = prepare_data(task_type, batch_size, task_params)

    rnn_hd = rnn_hidden_dim
    if match_rnn_hidden_dim_to_bottleneck_K: rnn_hd = bottleneck_K

    enc, forward_fn, data_train, loss_fn, train_cb_fn, _ = prepare(
      ponder_cost=ponder_cost, do_save=do_save,
      step_type=step_type, num_fixed_ticks=num_fixed_ticks, hps=hps,
      data_train=data_train, nc=nc, 
      input_dim=input_dim, rnn_hidden_dim=rnn_hd, eval_epoch=eval_epoch,
      bottleneck_K=bottleneck_K, beta=beta, M=M, 
      weight_normalization_type=weight_normalization_type, root_p=root_p,
      loss_fn_type=loss_fn_type, model_type=model_type, rnn_cell_type=rnn_cell_type,
      eps_halting=eps_halting, lambda_p=lambda_p, min_lambda_p=min_lambda_p,
      ponder_penalty_type=ponder_penalty_type, explore_weight=explore_weight, 
      accuracy_weight=accuracy_weight, cp_save_interval=cp_save_interval
    )

    if do_train and skip_existing:
      last_cp_f = train_cb_fn.get_save_p(num_train_epochs - 1)
      if os.path.exists(os.path.join(root_p, last_cp_f)): 
        print(f'Skipping: {last_cp_f}')
        continue

    if eval_epoch is not None:
      # evaluate
      hps['epoch'] = eval_epoch
      res_p = os.path.join(root_p, 'results')
      eval_ctx = EvaluateContext(save_p=res_p, do_save=do_save, default_batch_size=eval_batch_size)
      arg_sets.append((eval_ctx, enc, forward_fn, data_train, loss_fn, eval_epoch, hps))
    else:
      # train
      arg_sets.append((
        (i, len(its)), enc, forward_fn, data_train, loss_fn, train_cb_fn, 
        num_train_epochs, lr, seed, grad_clip))

  set_fn = train_set if eval_epoch is None else eval_set
  if not dry_run:
    run(set_fn, arg_sets, num_processes)
  else:
    print(f'Would run with: {len(arg_sets)} combinations')

# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':
  main()