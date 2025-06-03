import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.io import savemat
import numpy as np

from tasks import generate_logic_task_sequence, generate_addition_task_sequence

import os
from itertools import product

# ------------------------------------------------------------------------------------------------

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
    # x: (N x seq_len) | y: (seq_len,)
    return x, y

# ------------------------------------------------------------------------------------------------

class RecurrentClassifier(nn.Module):
  def __init__(
    self, *, input_dim: int, hidden_dim: int, rnn_cell_type: str, nc: int):
    
    assert rnn_cell_type in ['rnn', 'lstm']
    super().__init__()

    if rnn_cell_type == 'rnn':    self.rnn = nn.RNNCell(input_dim, hidden_dim)
    elif rnn_cell_type == 'lstm': self.rnn = nn.LSTMCell(input_dim, hidden_dim)
    else: assert False

    self.halt = nn.Linear(hidden_dim, 1)
    self.dec = nn.Sequential(
      nn.Linear(hidden_dim, nc),
      # nn.Tanh(),
    )
    self.rnn_cell_type = rnn_cell_type

  def fixed_ticks_step(self, sx: torch.Tensor, xi: torch.Tensor, num_ticks: int):
    sn = sx
    for _ in range(num_ticks):
      sn = self.rnn(xi, sn)
      h = sn[0] if self.rnn_cell_type == 'lstm' else sn
      y = self.dec(h)
    pt = torch.zeros((xi.shape[0],))
    yt = y
    sx = sn
    nt = torch.ones((xi.shape[0],), dtype=torch.int64) * num_ticks
    return sx, yt, pt, nt

  def act_step(self, sx: torch.Tensor, xi: torch.Tensor, eps_halting: float, M: int):
    sn = sx
    pt = 0. # cumuluative halting probability
    p_hist = []
    y_hist = []
    s_hist = []
    c_hist = []
    keep_processing = torch.ones((xi.shape[0],), dtype=bool)
    n = 0
    while keep_processing.any() and n < M:
      sn = self.rnn(xi, sn)
      h = sn[0] if self.rnn_cell_type == 'lstm' else sn
      pn = torch.sigmoid(self.halt(h)).squeeze(1)
      y = self.dec(h)
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
    return sx, yt, pt, nt

  def forward(
    self, x: torch.Tensor, sx: torch.Tensor = None, eps_halting = 1e-2, M = 14, 
    step_type: str = 'act', num_fixed_ticks: int = 6):
    assert step_type in ['act', 'fixed']

    T = x.shape[-1]
    P = []
    Y = []
    N = []
    for t in range(T):
      xi = x.select(-1, t)
      if step_type == 'act':
        sx, yt, pt, nt = self.act_step(sx, xi, eps_halting, M)
      elif step_type == 'fixed':
        sx, yt, pt, nt = self.fixed_ticks_step(sx, xi, num_fixed_ticks)
      else: assert False
      # append to sequence
      P.append(pt)
      Y.append(yt)
      N.append(nt)
    P = torch.stack(P, dim=-1)
    Y = torch.stack(Y, dim=-1)
    N = torch.stack(N, dim=-1)
    P = torch.sum(P, dim=-1)
    return Y, P, N

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
  loss_fn, epoch_cb, *, num_epochs: int, lr: float, random_seed: int):
  """"""
  torch.manual_seed(random_seed)
  optim = torch.optim.Adam([*enc.parameters()], lr=lr)

  for e in range(num_epochs):
    for xs, ys in data_train:
      y, p, n = foward_fn(enc, xs)
      err_rate = sequence_error_rate(ys, torch.argmax(torch.softmax(y, dim=1), dim=1))

      optim.zero_grad()
      L = loss_fn(y, ys, p)
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
  for xs, ys in data:
    y, p, n = foward_fn(enc, xs)
    yh = torch.argmax(torch.softmax(y, dim=1), dim=1)
    err_rate = sequence_error_rate(ys, yh)
    seq_acc = sequence_acc(ys, yh)
    L = loss_fn(y, ys, p)
    return {'acc': seq_acc.item(), 'err_rate': err_rate.item(), 'ticks': n.float().mean().item()}

def evaluate(enc: RecurrentClassifier, foward_fn, data: DataLoader, loss_fn, cp_fname: str, hps: dict):
  res = do_evaluate(enc, foward_fn, data, loss_fn)
  res['hps'] = hps
  savemat(os.path.join(os.getcwd(), 'results', cp_fname.replace('.pth', '.mat')), res)

# ------------------------------------------------------------------------------------------------

def prepare_data(task_type: str, batch_size: int):
  if task_type == 'logic':
    nc = 2
    num_ops = 10
    input_dim = num_ops * 10 + 2
    stride = 1
    ldp = {'batch_size': batch_size, 'seq_len': 3, 'num_ops': num_ops}
    data_train = DataLoader(LogicDataset(**ldp, num_samples=None), batch_size=batch_size)

  elif task_type == 'addition':
    max_num_digits = 5
    input_dim = max_num_digits * 10
    stride = 11
    nc = stride * (max_num_digits + 1)
    ldp = {'batch_size': batch_size, 'seq_len': 5, 'max_num_digits': max_num_digits}
    data_train = DataLoader(AdditionDataset(**ldp), batch_size=batch_size)

  else: assert False

  return data_train, input_dim, nc

def prepare(
  *, ponder_cost: float, do_save: bool, step_type: str, num_fixed_ticks: int, hps: dict,
  data_train: DataLoader, input_dim: int, nc: int, eval_epoch: int):
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
    rnn_cell_type=rnn_cell_type, nc=nc
  )

  def gen_cp_fname_fn(e: int): return f'act-checkpoint-{make_cp_id(hps)}-{e}.pth'
  def loss_fn(y, ys, p): return nn.functional.cross_entropy(y, ys) + ponder_cost * torch.mean(p)
  def forward_fn(enc, xs: torch.Tensor): 
    return enc(xs, step_type=step_type, num_fixed_ticks=num_fixed_ticks)
  def train_cb_fn(enc, e: int):
    if not do_save: return
    if e % cp_save_interval != 0: return
    cp_fname = gen_cp_fname_fn(e)
    cp = {'state': enc.state_dict(), 'hps': hps}
    torch.save(cp, os.path.join(os.getcwd(), 'data', cp_fname))

  if eval_epoch is not None:
    enc.eval()
    cp = torch.load(os.path.join(os.getcwd(), 'data', gen_cp_fname_fn(eval_epoch)))
    enc.load_state_dict(cp['state'])

  args = (enc, forward_fn, data_train, loss_fn, train_cb_fn, gen_cp_fname_fn)
  return args

# ------------------------------------------------------------------------------------------------

def main():
  eval_batch_size = 10_000

  seeds = [61, 62, 63]
  eval_epochs = list(np.arange(0, 14_001, 2_000))
  ponder_costs = [1e-3 * 0.5, 1e-3 * 2, 1e-3 * 3, 1e-3 * 4]

  # eval_epochs = [None]
  ponder_costs = [1e-3 * 1]

  its = product(seeds, eval_epochs, ponder_costs)
  for it in its:
    seed, eval_epoch, ponder_cost = it
    hps = {}

    task_type = hps['task_type'] = 'logic'
    train_batch_size = hps['batch_size'] = 32
    seed = hps['seed'] = seed
    # step_type = 'fixed'
    step_type = 'act'
    num_fixed_ticks = 6
    lr = 1e-3
    num_epochs = 15_000

    batch_size = train_batch_size if eval_epoch is None else eval_batch_size
    data_train, input_dim, nc = prepare_data(task_type, batch_size)

    enc, forward_fn, data_train, loss_fn, train_cb_fn, gen_cp_fname_fn = prepare(
      ponder_cost=ponder_cost, do_save=True,
      step_type=step_type, num_fixed_ticks=num_fixed_ticks, hps=hps,
      data_train=data_train, nc=nc, input_dim=input_dim, eval_epoch=eval_epoch
    )

    if eval_epoch is not None:
      cp_fname = gen_cp_fname_fn(eval_epoch)
      hps['epoch'] = eval_epoch
      evaluate(enc, forward_fn, data_train, loss_fn, cp_fname, hps)
    else:
      train(
        enc, forward_fn, data_train, loss_fn, train_cb_fn, 
        num_epochs=num_epochs, lr=lr, random_seed=seed)

# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':
  main()

# if stride > 1: y = y.view(y.shape[0] * num_classif, stride, y.shape[-1])
# if stride > 1: ys = ys.view((ys.shape[0] * num_classif, y.shape[-1]))
# if stride > 1: y = y[:, :, 1:]