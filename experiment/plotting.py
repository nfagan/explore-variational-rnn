from matplotlib.patches import Ellipse
from scipy.stats import chi2
import matplotlib.pyplot as plt
import numpy as np
import torch

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

def plot_embeddings(ax, mu: torch.Tensor, L: torch.Tensor, y: torch.Tensor, cmap: np.ndarray):
  assert mu.shape[1] == 2
  assert L.shape[1:] == (2, 2)
  assert mu.shape[0] == L.shape[0]
  cov = L @ L.transpose(-1, -2)
  # cov = cov.transpose(-1, -2)

  for i in range(mu.shape[0]):
    plot_gaussian_ellipse(
      mu[i, :].detach().cpu().numpy(), 
      cov[i, :, :].detach().cpu().numpy(), ax=ax, color=cmap[y[i], :], linewidth=0.5)