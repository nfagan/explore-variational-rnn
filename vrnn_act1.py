from tasks import (
  generate_logic_task_sequence, generate_addition_task_sequence, 
  generate_parity_task_sequence
)

from experiment.logictask import LogicDataset as LD

from models import (
  RecurrentVariationalPredictor, SimpleRecurrentPredictor,
  variational_forward, variational_loss, simple_forward, simple_loss
)

from plotting import PlotContext, plot_line, plot_lines, plot_scatter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
import os, re

# CP_PREFIX = 'variational-beta_2'  # 3 ticks
# CP_PREFIX = 'variational-high-beta'   # beta = 8, 3 ticks
# CP_PREFIX = 'variational-beta_0.1_5_ticks'      
# CP_PREFIX = 'variational-beta_0.1_4_ticks'
CP_PREFIX = 'variational-beta_0.1_16_ticks'
# CP_PREFIX = 'variational-low-beta'      # beta = 0.1, 3 ticks
# CP_PREFIX = 'simple-large-logic'
  
@dataclass
class TrainContext:
  save_checkpoints: bool
  root_p: str

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
      # x, y, m = gen_addition(self.seq_len, self.max_num_digits)
    m = torch.ones_like(y)
    # x: (seq_len x N) | y: (seq_len x 1) | m: (seq_len x 1)
    return x, y.type(torch.long), m

class AdditionDataset(Dataset):
  def __init__(self, batch_size: int, seq_len: int, max_num_digits: int):
    super().__init__()
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.max_num_digits = max_num_digits
  
  def __len__(self):
    return self.batch_size
  
  def __getitem__(self, idx):
    x, y, m = generate_addition_task_sequence(self.seq_len, self.max_num_digits)
    return x, y, m

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
    return x.unsqueeze(0), y.unsqueeze(0).type(torch.long), mask.unsqueeze(0)

def eval_model(model, forward_fn, loss_fn, data_loader, verbose=True):
  model.eval()
  total_loss = total_acc = 0.0
  total_outs = []
  ns = 0
  for x_seq, y_seq, mask in data_loader:
    loss, acc, outs = forward_fn((model, loss_fn, x_seq, y_seq, mask))
    total_loss += loss.item()
    total_acc += acc.item()
    total_outs.append(outs)
    ns += 1
  avg_loss = total_loss / ns
  avg_acc = total_acc / ns
  if verbose: print(f'(Eval): Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}')
  return avg_loss, avg_acc, total_outs

def cp_epochs(root_p: str, prefix: str) -> List[str]:
  def find_and_extract(directory: str, substring: str):
    results = []
    # Compile regex to capture the last integer before the ".pth" extension.
    pattern = re.compile(r'(\d+)(?=\.pth$)')
    
    # Loop over files in the directory.
    for filename in os.listdir(directory):
      if filename.endswith('.pth') and substring in filename:
        match = pattern.search(filename)
        if match:
          number = int(match.group(1))
          results.append((filename, number))
    return results
  res = find_and_extract(os.path.join(root_p, 'checkpoints'), prefix)
  res = sorted(res, key=lambda x: x[1])
  res = [v[1] for v in res]
  return res

def cp_fname(root_p: str, prefix: str, epoch: int):
  return os.path.join(root_p, 'checkpoints', f'cp-{prefix}-{epoch}.pth')

def train_model(
  context: TrainContext, model, train_forward_fn, eval_forward_fn, loss_fn, train_data, eval_data, 
  num_epochs: int, learning_rate=1e-3):
  """
  """
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
  for epoch in range(num_epochs):
    model.train()

    total_loss = total_acc = 0.0
    ns = 0
    for x_seq, y_seq, mask in train_data:
      optimizer.zero_grad()
      loss, acc, _ = train_forward_fn((model, loss_fn, x_seq, y_seq, mask))
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
      total_acc += acc.item()
      ns += 1

    avg_loss = total_loss / ns
    avg_acc = total_acc / ns

    if epoch % 100 == 0 or epoch + 1 == num_epochs:
      p_done = (epoch+1) / num_epochs * 100
      print(f'Epoch {epoch+1} of {num_epochs} ({p_done:.2f}%) | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}')

    if epoch % 1000 == 0 or epoch + 1 == num_epochs:
      eval_model(model, eval_forward_fn, loss_fn, eval_data)
      if context.save_checkpoints:
        sd = {'state': model.state_dict()}
        torch.save(sd, cp_fname(context.root_p, CP_PREFIX, epoch))

def do_eval_model(model, prefix, base_forward_fn, loss_fn, eval_data):  
  def has_term(outs, name: str):
    return len(outs) > 0 and outs[0] and name in outs[0][0]
  
  context = PlotContext(subdir=prefix)

  num_ticks = []
  accs = []
  outs = []
  for nt in range(16):
    eval_forward_fn = lambda args: base_forward_fn(*args, forced_num_ticks=nt + 1)
    loss, acc, out = eval_model(model, eval_forward_fn, loss_fn, eval_data)
    accs.append(acc)
    outs.append(out)
    num_ticks.append(nt + 1)

  accs = np.array(accs)
  num_ticks = np.array(num_ticks)
  plot_line(num_ticks, accs, 'ticks', 'accuracy', context=context, ylim=[0.5, 1])

  if has_term(outs, 'mutual_information'):
    mis = [np.mean(np.array([y['mutual_information'] for y in x])) for x in outs]
    plot_line(num_ticks, mis, 'ticks', 'mutual information', context=context)

  if has_term(outs, 'kl_divergence'):
    kls = [np.mean(np.array([y['kl_divergence'].item() for y in x])) for x in outs]
    plot_line(num_ticks, kls, 'ticks', 'KL(z, N(0, 1))', context=context)

if __name__ == '__main__':
  task = 'logic'
  # model_type = 'variational'
  model_type = 'simple' if 'simple' in CP_PREFIX else 'variational'
  
  assert task in ['addition', 'parity', 'logic']
  assert model_type in ['simple', 'variational']

  do_train = False
  train_batch_size = 128
  # train_batch_size = 1024
  eval_batch_size = 1000 * 10
  num_epochs = 10000 * 2
  # num_epochs = 100
  max_num_ticks = 8
  variational_latent_dim = 4
  ctx = TrainContext(
    save_checkpoints=True,
    root_p=os.getcwd()
  )
  
  beta = 0.1
  # beta = 8.
  # beta = 2.
  # beta = 1.

  if task == 'parity':
    input_dim = 64
    num_classes = 2
    rnn_type = 'rnn'
    rnn_hidden_dim = 128
    train_data = ParityDataset(train_batch_size, input_dim)
    eval_data = ParityDataset(eval_batch_size, input_dim)

  elif task == 'logic':
    num_ops = 10
    num_classes = 2
    rnn_type = 'gru'
    rnn_hidden_dim = 128
    input_dim = num_ops * 10 + 2
    seq_len = 3
    if False:
      ldp = {'seq_len': seq_len, 'num_operations': num_ops, 
            'per_position_targets': True, 'match_non_sequence_tasks': True}
      train_data = LD(**ldp, num_samples=10000)
      eval_data = LD(**ldp, num_samples=5000)
    else:
      train_data = LogicDataset(train_batch_size, seq_len, num_ops)
      eval_data = LogicDataset(eval_batch_size, seq_len, num_ops)

  elif task == 'addition':
    num_digits = 5
    input_dim = num_digits * 10
    num_classes = 11 * (num_digits + 1)
    rnn_type = 'lstm'
    rnn_hidden_dim = 512
    train_data = AdditionDataset(train_batch_size, 5, num_digits)
    eval_data = AdditionDataset(eval_batch_size, 5, num_digits)

  else: assert False

  train_data = DataLoader(train_data, batch_size=train_batch_size)
  eval_data = DataLoader(eval_data, batch_size=eval_batch_size)

  if model_type == 'simple':
    model = SimpleRecurrentPredictor(
      input_dim=input_dim, hidden_dim=rnn_hidden_dim, output_dim=num_classes, 
      max_num_ticks=max_num_ticks, rnn_type=rnn_type, last_tick_only=True, last_step_only=True)
    loss_fn = lambda args: simple_loss(*args)
    forward_fn = simple_forward

  elif model_type == 'variational':
    model = RecurrentVariationalPredictor(
      input_dim=input_dim, hidden_dim=rnn_hidden_dim, latent_dim=variational_latent_dim, 
      output_dim=num_classes, max_num_ticks=max_num_ticks)
    loss_fn = lambda args: variational_loss(*args, beta=beta)
    forward_fn = variational_forward

  else: assert False

  train_forward_fn = lambda args: forward_fn(*args, forced_num_ticks=max_num_ticks)
  eval_forward_fn = lambda args: forward_fn(*args, forced_num_ticks=max_num_ticks)
  
  if do_train:
    train_model(
      ctx, model, train_forward_fn, eval_forward_fn, loss_fn, train_data, eval_data, 
      num_epochs=num_epochs, learning_rate=1e-3)
    
  else:
    if False:
      eps = np.array(cp_epochs(ctx.root_p, CP_PREFIX))
      accs = np.zeros(eps.shape)
      for i, ep in enumerate(eps):
        model.load_state_dict(torch.load(cp_fname(ctx.root_p, CP_PREFIX, ep))['state'])
        loss, acc, out = eval_model(model, eval_forward_fn, loss_fn, eval_data)
        accs[i] = acc

      plot_line(eps, accs, 'episodes', 'accuracy_over_training', 
                context=PlotContext(subdir=CP_PREFIX), ylim=[0.4, 1.])
    
    if True:
      make_fn = lambda tick: lambda args: forward_fn(*args, forced_num_ticks=tick)

      ticks = [1, 2, 3, 4, 5, 8, 16]
      # ticks = ticks[:2]
      ticks_str = [f'num_ticks={x}' for x in ticks]
      eps = np.array(cp_epochs(ctx.root_p, CP_PREFIX))
      # eps = eps[:3]

      last_only = True
      shp = (1, len(ticks)) if last_only else (len(eps), len(ticks))
      max_mis = np.zeros(shp)
      max_mis = np.zeros(shp)
      max_mi_ticks = np.zeros_like(max_mis)
      max_mi_src_eps = np.zeros_like(max_mis)
      accs = np.zeros((len(eps), len(ticks)))

      for ti, t in enumerate(ticks):
        print(f'{ti+1} of {len(ticks)}')
        for i, ep in enumerate(eps):
          print(f'\t{i+1} of {len(eps)}')

          prefix = f'variational-beta_0.1_{t}_ticks'
          # @TODO, change prefix for forced_num_ticks=3 for consistency
          prefix = 'variational-low-beta' if t == 3 else prefix 
          model.load_state_dict(torch.load(cp_fname(ctx.root_p, prefix, ep))['state'])

          if False:
            loss, acc, out = eval_model(model, make_fn(t), loss_fn, eval_data, verbose=False)
            accs[i, ti] = acc
          
          if not last_only or i + 1 == len(eps):
            res = [eval_model(model, make_fn(tick), loss_fn, eval_data, verbose=False) for tick in ticks]
            mis = [x[2][0]['mutual_information'].mean() for x in res]
            di = 0 if last_only else i
            max_mis[di, ti] = max(mis)
            max_mi_ticks[di, ti] = ticks[np.argmax(mis)]
            max_mi_src_eps[di, ti] = ep

      do_norm = lambda x: (x - x.min()) / (1. if x.max() == x.min() else x.max() - x.min())
      summary_ctx = PlotContext(subdir='summary')
      plot_scatter(max_mi_ticks, max_mis, ticks_str,  do_norm(max_mi_src_eps), 
                   'best ticks', 'best mutual information', context=summary_ctx, scale=[8, 8])
      # plot_lines(eps, accs, ticks_str, 'episodes', 'accuracy_over_training', context=summary_ctx)

    if False:
      eval_epoch = max(eps)
      model.load_state_dict(torch.load(cp_fname(ctx.root_p, CP_PREFIX, eval_epoch))['state'])
      do_eval_model(model, CP_PREFIX, forward_fn, loss_fn, eval_data)