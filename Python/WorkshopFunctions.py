import types
import numpy as np
import matplotlib.pyplot as plt


def GenerateData(f, sig_eps, N, plot_data = False, ax_lims = None, ax = None):
    t = np.random.default_rng().normal(scale=1, size=(N,1))
    y = f(t) + np.random.default_rng().normal(scale=sig_eps, size=(N,1))

    if plot_data:
        if ax is None:
            fig, ax = plt.subplots() # Create a new figure for plotting

        ax.plot(t, y, 'ko', linewidth=1.5, markerfacecolor="#BBBBBB", markersize=4)
        if not(ax_lims is None):
            ax.set_xlim(left = ax_lims[0], right = ax_lims[1])
            ax.set_ylim(bottom = ax_lims[2], top = ax_lims[3])
        return types.SimpleNamespace(t=t, y=y), ax

    else:
        return types.SimpleNamespace(t = t, y = y)