from matplotlib import pyplot as plt
import numpy as np
from dataclasses import dataclass
import os

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

def plot_scatter(xs, ys, by, s, xlab, ylab, context: PlotContext=PlotContext(), ylim=None, scale=[8, 4]):
  f = plt.figure(1)
  plt.clf()
  h = []
  ss = s * scale[0] + scale[1]
  for col in range(xs.shape[1]): h.append(plt.scatter(xs[:, col], ys[:, col], ss[:, col]))
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