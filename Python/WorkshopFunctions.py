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

def ReadProcessData(filename):
    RawData = np.genfromtxt(filename, delimiter=',')
    Data = types.SimpleNamespace(t  = RawData[:, 0],
                                 u1 = RawData[:, 1],
                                 u2 = RawData[:, 2],
                                 y  = RawData[:, 3])
    Data.t[0] = 0
    return Data


def CreateLaggedDesignMatrix(TimeData, L, f = 1):
    N = np.rint(f*len(TimeData.t)).astype(int)
    td = TimeData.__dict__
    X = np.zeros([N-(L+1), 3*(L+1)])
    name = ['y', 'u1', 'u2']
    for j in range(3):
        for i in range(L+1):
            X[:, j*(L+1) + i] = td[name[j]][(L-i):N-(i+1)]

    y = TimeData.y[L+1:N]
    return X, y

def PredictTimeSeries(mdl, TimeData, L):
    N = len(TimeData.t)
    y = np.zeros([N,1])
    X = CreateLaggedDesignMatrix(TimeData, L, f = 1)[0]
    for i in range(N-L-1):
        y_lag = y[i:(i+L+1)]
        yX = np.concatenate((y_lag.transpose(), X[np.newaxis, i, L:-1]), axis = 1)
        project = np.matmul(yX, mdl.Q)
        y[L+i] = np.matmul(project, mdl.coef)

    return y


