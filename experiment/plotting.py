from matplotlib.patches import Ellipse
from scipy.stats import chi2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from dataclasses import dataclass

@dataclass
class PlotContext:
  show_plot: bool = False
  save_p: str = os.path.join(os.getcwd(), 'plots')
  subdir: str = None
  # save_p: str = None

  def full_p(self, req: bool=False):
    res = self.save_p if self.subdir is None else os.path.join(self.save_p, self.subdir)
    if req: os.makedirs(res, exist_ok=True)
    return res

def panels(N: int):
  return [plt.subplot(*subplot_shape(N), i+1) for i in range(N)]

def subplot_shape(N: int):
  if N <= 3:
    return [1, N]
  elif N == 8:
    return [2, 4]
  else:
    n_rows = round(np.sqrt(N))
    n_cols = np.ceil(N/n_rows)
    return [int(n_rows), int(n_cols)]
  
def maybe_save_fig(fname: str, context: PlotContext=PlotContext(), f=None):
  if f is None: f = plt.gcf()
  if context.save_p is not None: f.savefig(os.path.join(context.full_p(True), f'{fname}.png'))

def plot_line(xs, ys, xlab, ylab, context: PlotContext=PlotContext(), ylim=None):
  f = plt.figure(1)
  plt.clf()
  plt.plot(xs, ys)
  plt.xlabel(xlab)
  plt.ylabel(ylab)
  if ylim is not None: plt.ylim(ylim)
  if context.show_plot: plt.show()
  plt.draw()
  if context.save_p is not None: f.savefig(os.path.join(context.full_p(True), f'{ylab}.png'))

def plot_lines(xs, ys, by, xlab, ylab, context: PlotContext=PlotContext(), ylim=None):
  f = plt.figure(1)
  plt.clf()
  h = plt.plot(xs, ys)
  plt.xlabel(xlab)
  plt.ylabel(ylab)
  num_lines = len(by)
  cmap = plt.get_cmap('spring', num_lines)(np.arange(num_lines))[:, :3]
  for i in range(num_lines):
    h[i].set_color(cmap[i, :])
    h[i].set_label(f'{by[i]}')
  if ylim is not None: plt.ylim(ylim)
  plt.legend()
  if context.show_plot: plt.show()
  plt.draw()
  if context.save_p is not None: f.savefig(os.path.join(context.full_p(True), f'{ylab}.png'))

def plot_image(ax, x, y, C, **imshow_kwargs):
  x = np.asarray(x)
  y = np.asarray(y)

  # imshow expects the *edges* of the image, so we infer them from the centers
  dx = (x[1] - x[0]) if len(x) > 1 else 1.0
  dy = (y[1] - y[0]) if len(y) > 1 else 1.0
  extent = [x[0] - dx/2, x[-1] + dx/2,
            y[0] - dy/2, y[-1] + dy/2]

  img = ax.imshow(
    C,
    extent=extent,
    origin="lower",     # y increases upward, matching MATLAB’s imagesc
    aspect="auto",
    **imshow_kwargs,
  )
  return img

def plot_gaussian_ellipse(mu, cov, ax=None, confidence=0.95, **ellipse_kwargs):
  mu = np.asarray(mu)
  cov = np.asarray(cov)

  # χ² quantile for the desired confidence and 2 d.o.f.
  chi2_val = chi2.ppf(confidence, df=2)

  # Eigen‑decomposition → principal axes
  eigvals, eigvecs = np.linalg.eigh(cov)
  width, height = 2 * np.sqrt(eigvals * chi2_val)      # full lengths
  width, height = height, width
  angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
  # angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

  if ax is None: fig, ax = plt.subplots()

  # width = 1.
  # height = 2.
  # angle = 0.

  ellipse = Ellipse(
    xy=mu,
    width=width,
    height=height,
    angle=angle,
    fill=False,            # no fill so we don’t pick a specific color
    **ellipse_kwargs,
  )
  ax.add_patch(ellipse)
  # ax.scatter(*mu, marker="x")      # mark the mean
  ax.set_aspect("equal", adjustable="datalim")
  ax.autoscale_view()

def plot_embeddings(ax, mu: torch.Tensor, cov: torch.Tensor, y: torch.Tensor, cmap: np.ndarray):
  assert mu.shape[1] == 2
  assert cov.shape[1:] == (2, 2)
  assert mu.shape[0] == cov.shape[0]
  # cov = cov.transpose(-1, -2)

  for i in range(mu.shape[0]):
    plot_gaussian_ellipse(
      mu[i, :].detach().cpu().numpy(), 
      cov[i, :, :].detach().cpu().numpy(), ax=ax, color=cmap[y[i], :], linewidth=0.5)