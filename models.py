import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.feature_selection

class SimpleRecurrentPredictor(nn.Module):
  def __init__(
    self, *, input_dim: int, hidden_dim: int, output_dim: int, max_num_ticks: int, rnn_type: str,
    last_step_only: bool, last_tick_only: bool):
    """
    """
    assert rnn_type in ['rnn', 'lstm', 'gru']
    super(SimpleRecurrentPredictor, self).__init__()
    self.hidden_dim = hidden_dim

    if rnn_type == 'rnn':     self.rnn_cell = nn.RNNCell(input_dim, hidden_dim)
    elif rnn_type == 'lstm':  self.rnn_cell = nn.LSTMCell(input_dim, hidden_dim)
    elif rnn_type == 'gru':   self.rnn_cell = nn.GRUCell(input_dim, hidden_dim)
    else: assert False

    self.fc_pred = nn.Linear(hidden_dim, output_dim)
    self.max_num_ticks = max_num_ticks
    self.rnn_type = rnn_type
    self.last_step_only = last_step_only
    self.last_tick_only = last_tick_only

  def forward(
    self, x_seq, y_seq, seq_mask, forced_num_ticks=None):
    """
    """
    batch_size, seq_len, _ = x_seq.size()
    
    logits_seq = []
    y_res = []
    ok = []
    nticks = []

    sm = seq_mask.detach()
    
    state = None
    for t in range(seq_len):
      x_t = x_seq[:, t, :]  # current input xₜ

      if forced_num_ticks is not None:
        nt = torch.ones((batch_size,), dtype=torch.long) * forced_num_ticks
      else:
        nt = 1 + torch.randint(0, self.max_num_ticks, (batch_size,))
      num_ticks = 0

      while True:
        not_finished = num_ticks < nt
        num_ticks += 1
        if not torch.any(not_finished): break

        state = self.rnn_cell(x_t, state)
        hx = state[0] if self.rnn_type == 'lstm' else state
        logits_t = self.fc_pred(hx)

        mask = not_finished[:, None].detach()
        mask = mask * sm[:, t, :]

        logits_t *= mask

        mask_last = True
        if self.last_step_only: mask_last = t + 1 == seq_len
        if self.last_tick_only: mask_last = mask_last & (num_ticks == nt).detach()
        
        logits_seq.append(logits_t)
        y_res.append(y_seq[:, t])
        ok.append(mask.squeeze(1).type(torch.bool) * mask_last)
        nticks.append(nt)
    
    # Stack time steps back into tensors.
    logits_seq = torch.stack(logits_seq, dim=1)       # [B, T, output_dim]
    y_res = torch.stack(y_res, dim=1)
    ok = torch.stack(ok, dim=1)    
    nticks = torch.stack(nticks, dim=1)
    return logits_seq, y_res, ok

class RecurrentVariationalPredictor(nn.Module):
  def __init__(
      self, *, input_dim: int, hidden_dim: int, 
      latent_dim: int, output_dim: int, max_num_ticks: int):
    super(RecurrentVariationalPredictor, self).__init__()
    self.hidden_dim = hidden_dim
    self.max_num_ticks = max_num_ticks

    self.rnn_cell = nn.LSTMCell(input_dim, hidden_dim)
    self.fc_mu = nn.Linear(hidden_dim, latent_dim)
    self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    # self.fc_pred = nn.Linear(hidden_dim + latent_dim, output_dim)
    self.fc_pred = nn.Linear(latent_dim, output_dim)
    self.fc_halt = nn.Linear(hidden_dim + latent_dim, 1)
  
  def reparameterize(self, mu, logvar):
    """Applies the reparameterization trick to sample z ~ q(z|x)"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
  
  def forward(self, x_seq, y_seq, seq_mask, forced_num_ticks=None):
    batch_size, seq_len, _ = x_seq.size()
      
    # Initialize LSTM hidden and cell states.
    hx = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
    cx = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)

    sm = seq_mask.detach()
    
    logits_seq = []
    mu_seq = []
    logvar_seq = []
    z_seq = []
    y_res = []
    ok = []
    nticks = []
    
    for t in range(seq_len):
      x_t = x_seq[:, t, :]  # current input xₜ

      if forced_num_ticks is not None:
        nt = torch.ones((batch_size,), dtype=torch.long) * forced_num_ticks
      else:
        nt = 1 + torch.randint(0, self.max_num_ticks, (batch_size,))
      num_ticks = 0

      while True:
        not_finished = num_ticks < nt
        num_ticks += 1
        if not torch.any(not_finished): break

        hx, cx = self.rnn_cell(x_t, (hx, cx))
        
        # Compute latent distribution parameters from hx.
        mu_t = self.fc_mu(hx)
        logvar_t = self.fc_logvar(hx)
        z_t = self.reparameterize(mu_t, logvar_t)

        # print('tick: ', num_ticks, torch.mean(torch.mean(logvar_t.exp(), dim=0)))
        # print('tick: ', num_ticks, torch.mean(torch.mean(mu_t, dim=0)))
        
        # Combine hidden state and latent sample for target prediction.
        # pred = torch.cat([hx, z_t], dim=1)
        pred = z_t.clone()
        logits_t = self.fc_pred(pred)

        # # Halt
        # halt = torch.sigmoid(self.fc_halt(pred))
        mask = not_finished[:, None].detach()
        mask = mask * sm[:, t, :]

        logits_t *= mask
        mu_t *= mask
        logvar_t *= mask
        z_t *= mask
        
        # Store outputs for each time step.
        logits_seq.append(logits_t)
        mu_seq.append(mu_t)
        logvar_seq.append(logvar_t)
        z_seq.append(z_t)
        y_res.append(y_seq[:, t])
        ok.append(mask.squeeze(1).type(torch.bool))
        nticks.append(nt)
    
    # Stack time steps back into tensors.
    logits_seq = torch.stack(logits_seq, dim=1)       # [B, T, output_dim]
    mu_seq = torch.stack(mu_seq, dim=1)               # [B, T, latent_dim]
    logvar_seq = torch.stack(logvar_seq, dim=1)       # [B, T, latent_dim]
    z_seq = torch.stack(z_seq, dim=1)                 # [B, T, latent_dim]
    y_res = torch.stack(y_res, dim=1)
    ok = torch.stack(ok, dim=1)    
    nticks = torch.stack(nticks, dim=1)

    return logits_seq, mu_seq, logvar_seq, z_seq, y_res, ok

def accuracy(logits_seq, targets, ok):
  lf, tf = reshape_outputs(logits_seq, targets, ok)
  idx = torch.argmax(torch.softmax(lf, dim=1), dim=1)
  acc = (tf == idx).sum() / tf.shape[0]
  return acc

def reshape_outputs(logits_seq, targets, ok):
  logits_flat = logits_seq.flatten(0, 1)      # [B*T, output_dim]
  targets_flat = targets.flatten(0, 1)        # [B*T]
  ok_flat = ok.view(-1)
  lf = logits_flat[ok_flat, :]
  tf = targets_flat[ok_flat, :]
  assert lf.shape[1] % tf.shape[1] == 0
  B = lf.shape[0]
  N = lf.shape[1] // tf.shape[1]
  D = logits_seq.shape[2]//N
  lff = lf.view(lf.shape[0], D, N).view(B * D, N)
  tff = tf.view(B * D)
  return lff, tff

def variational_loss(logits_seq, targets, mu_seq, logvar_seq, ok, beta=1.0):
  batch_size, seq_len, output_dim = logits_seq.size()
  
  # Flatten predictions and targets to compute cross entropy.
  num_ok = ok.sum()
  lf, tf = reshape_outputs(logits_seq, targets, ok)
  ce_loss = F.cross_entropy(lf, tf, reduction='sum')
  
  # KL divergence term (against standard normal) per time step.
  kl_div = -0.5 * torch.sum(1 + logvar_seq - mu_seq.pow(2) - logvar_seq.exp())
  mu_kl_div = kl_div/num_ok

  outs = {'kl_divergence': mu_kl_div}
  
  return ce_loss/num_ok + beta * mu_kl_div, outs

def variational_forward(model: RecurrentVariationalPredictor, loss_fn, x_seq, y_seq, mask, **kwargs):
  logits_seq, mu_seq, logvar_seq, z_seq, dst_seq, ok = model(x_seq, y_seq, mask, **kwargs)

  outs = {}
  if True:
    with torch.no_grad():
      zf, tf = reshape_outputs(z_seq, dst_seq, ok)
      mi = sklearn.feature_selection.mutual_info_classif(zf.cpu().numpy(), tf.cpu().numpy())
      outs['mutual_information'] = mi

  loss, loss_outs = loss_fn((logits_seq, dst_seq, mu_seq, logvar_seq, ok))
  acc = accuracy(logits_seq, dst_seq, ok)

  outs.update(loss_outs)

  return loss, acc, outs

def simple_loss(logits_seq, targets, ok):
  batch_size, seq_len, output_dim = logits_seq.size()
  num_ok = ok.sum()
  lf, tf = reshape_outputs(logits_seq, targets, ok)
  ce_loss = F.cross_entropy(lf, tf, reduction='sum')
  outs = {}
  return ce_loss/num_ok, outs

def simple_forward(model: SimpleRecurrentPredictor, loss_fn, x_seq, y_seq, mask, **kwargs):
  logits_seq, dst_seq, ok = model(x_seq, y_seq, mask, **kwargs)
  loss, loss_outs = loss_fn((logits_seq, dst_seq, ok))
  acc = accuracy(logits_seq, dst_seq, ok)
  outs = {}
  outs.update(loss_outs)
  return loss, acc, outs