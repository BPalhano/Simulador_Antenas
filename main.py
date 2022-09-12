import numpy as np
from numpy import matlib
from matplotlib import pyplot as plt

#   Global variables

nUe = 8     # Number of users in square
L = 400     # Square Length
nAnt = 64   # Number of Antennas per AP
c = 299792458   # Light Speed
freq = 3000000000   # 3 * 10^{9}. Hz
loop = 1000

#  Setup the APs:

APperdim = np.sqrt(nAnt)
APindex = np.linspace(L/APperdim, L, int(APperdim)) - (L/APperdim) / 2
APcellfree = matlib.repmat(APindex, int(APperdim), 1) + 1j * (matlib.repmat(APindex, int(APperdim), 1)).T
UEloc = (np.random.rand(loop, nUe) + 1j * np.random.rand(loop, nUe)) * L

#   Main containers:
SINR_cellfree = np.array([])
SNR_total = np.array([])

#   Funções


def pow2db(x, size):

    db = x.copy()
    for i in range(size):
        db[i] = 10 * np.log10(db[i] / 10)

    return db


def computeSINRs_MMSE(channel):

    vector = channel.copy()
    r = channel.shape[0]
    c = channel.shape[1]

    SINRs = np.array([])

    vectimesT = vector @ vector.T
    vector = vector.T

    for k in range(c):

        desiredChannel = vector[k]
        desiredtimesT = desiredChannel @ desiredChannel.T
        solved = np.linalg.lstsq((vectimesT - desiredtimesT + np.eye(r, dtype=float)), desiredChannel , rcond=-1)[0]
        temp = desiredChannel @ solved
        SINRs = np.hstack((SINRs, temp.reshape(-1)))


    return SINRs


def SNR(dist):
    Sig = 10+96-30.5 - 37.7*np.log10(np.sqrt(np.square(dist) + 100))

    return Sig


def db2pot(x):
    pot = np.array([])
    for i in range(len(x)):
        pot = np.hstack((pot, 10**(x[i] / 10)))

    return pot

#   Main loop:


for i in range(loop):

    channelCellfree = np.array([])

    for j in range(nUe):

        distanceCellfree = np.absolute(APcellfree.reshape(-1) - UEloc[i][j])
        channelEvaluate = np.sqrt(np.multiply(np.exp(1j * 2 * np.pi * np.random.rand(nAnt, 1)), SNR(distanceCellfree)))
        channelEvaluate = np.diagonal(channelEvaluate)
        channelCellfree = np.hstack((channelCellfree, channelEvaluate.T))

    channelCellfree = channelCellfree.reshape(nAnt, nUe)
    SINR_cellfree = np.hstack((SINR_cellfree,computeSINRs_MMSE(channelCellfree)))

#   ploting:

plt.plot(pow2db(sorted(SINR_cellfree),(nUe*loop) ), np.linspace(0,1, (nUe*loop)), color='red', linestyle='dashed')
plt.xlabel('SINR [dB]')
plt.ylabel('CDF')
plt.title('eCDF of Cell-free Massive MIMO model.')
plt.show()


