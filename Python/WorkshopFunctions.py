import types
import numpy as np
import matplotlib.pyplot as plt


def GenerateData(f, sig_eps, N, ax = None, ax_lims = None):
    t = np.random.default_rng().normal(scale=1, size=(N,1))
    y = f(t) + np.random.default_rng().normal(scale=sig_eps, size=(N,1))

    if not ax is None:
        ax.plot(t, y, 'ko', linewidth=1.5, markerfacecolor="#BBBBBB", markersize=4)
        if not(ax_lims is None):
            ax.set_xlim(left = ax_lims[0], right = ax_lims[1])
            ax.set_ylim(bottom = ax_lims[2], top = ax_lims[3])

    return types.SimpleNamespace(t=t, y=y)


def CreateGaussDesignMatrix(t, c, s = None):
    if s is None:
        s = np.sqrt(np.mean(c[1:-1] - c[0:-2]))

    X = np.zeros([len(t), len(c)])
    for i in range(len(c)):
        X[:,i] =  np.exp( -(t-c[i])**2/s).flatten()

    return X