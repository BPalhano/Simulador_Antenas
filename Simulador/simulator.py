################################################ BIBLIOTECAS ###########################################################
import numpy as np
import pandas as pd
import seaborn as sbn
from scipy import stats as scs
from matplotlib import pyplot as plt
from numpy import random
import math
########################################################################################################################

############################################  VARIÁVEIS GLOBAIS ########################################################
N = int(input('Numero de usuários: '))
R = 200
loop = int(input('Quantos laços de repetição: '))

print('\n\n')

c = 299792458
freq = 6000000000
_lambda = c / freq
loop = range(loop)
it = range(N + 1)
SINR_final = np.array([])
SINR_final2 = np.array([])

########################################################################################################################

# #################################### CONVERSOR DE UNIDADES ###########################################################
def dB_to_linear(x):  # conversor de dB para escala linear

    x = 10 ** (x / 10)

    return x  # retorno em escala linear


def linear_to_dB(x):  # conversor de escala linear para dB

    x = 10 * np.log10(x)

    return x  # retorno em dB


def Soma(vet1):  # Função para somar todas os elementos de um vetor

    s = 0

    for i in vet1:
        s += i

    return s
########################################################################################################################


# #################################### GERADORES DE POSICIONAMENTO #####################################################
def ERB_TM(num, dist):
    ERB = np.array([0, 0, 100])
    ap = dist * math.sqrt(3)/2
    TM = np.array([R * random.uniform(0, 1), R * random.uniform(0, 1), 0])

    contador_ERB = 0

    while contador_ERB != 6:
        arr1 = np.array(
            [2*ap * math.cos((math.pi / 6) + contador_ERB*math.pi/3), 2*ap * math.sin((math.pi / 6) + contador_ERB*math.pi/3), 100])
        ERB = np.vstack((ERB, arr1))

        arr2 = np.array(
            [arr1[0] + dist * random.uniform(0, 1), arr1[1] + dist * random.uniform(0, 1), 100 * random.uniform(-1, 0)])
        TM = np.vstack((TM, arr2))

        contador_ERB += 1

    if N > contador_ERB:
        contador_TM = 0

        while contador_TM != num:
            aux = np.array([6 * dist * random.normal(0, 1), 6 * dist * math.sin(math.pi / 3) * random.uniform(0, 1),
                            100 * random.uniform(0, 1)], dtype=object)

            TM = np.vstack((TM, aux))
            contador_TM += 1

    return ERB, TM


def matriz_dist(vet1, vet2):  # matriz de distâncias de ERB i para TM j

    d = np.array([])

    for i in vet1:

        for j in vet2:
            dist = math.sqrt(pow((i[0] - j[0]), 2) + pow((i[1] - j[1]), 2) + pow((i[2] - j[2]), 2))
            d = np.hstack((d, dist))

    return d  # retorna uma matriz i x j
########################################################################################################################


# ################################# GERADORES DE RUIDO #################################################################


def Path_loss(vet1):  # Gerador de interferência linear.

    vet1 = vet1.copy()

    for i in range(len(vet1)):
        vet1[i] = dB_to_linear(20 * np.log10(4 * math.pi * vet1 / _lambda))

    return vet1  # em W


def Sombreamento_teste(vet1, sd):
    aux = vet1.copy()

    rv = random.lognormal(mean=0, sigma=sd)

    while rv > 100:
        rv = random.lognormal(mean=0, sigma=sd)

    for i in range(len(vet1)):
        aux[i] = dB_to_linear(rv)

    return aux  # em W

def Sombreamento(vet1, sd):

    aux = vet1.copy()

    for i in range(len(vet1)):
        rv = random.lognormal(mean=0, sigma=sd)

        while(rv > 100):
            rv = random.lognormal(mean=0, sigma=sd)

        aux[i] = dB_to_linear(rv)
    return aux #  em W

def Fast_fadding(vet1):
    aux = vet1.copy()

    for i in range(len(vet1)):
        rv = scs.rice.rvs(10)
        aux[i] = dB_to_linear(rv)

    return aux  # em W


def Ganho(vet1, vet2, vet3, K):  # Ganho total do sistema

    G = vet1.copy()

    for i in range(len(vet1)):
        G[i] = vet1[i] * vet2[i] * vet3[i] * K

    return G  # em W


def SINR(vet1, const):
    matrix = vet1.copy()
    trace = 0

    for i in range(0, 7):
        trace += matrix[7 * i]

    tot = Soma(matrix)

    cnt = tot - trace

    SNR = np.array([])

    for i in range(7):
        val = matrix[i * 7] / (const + cnt)

        SNR = np.hstack((SNR, val))

    return SNR


def sbn_eCDF(vet1):
    dataset = vet1.copy()
    df = pd.DataFrame(dataset)

    sbn.ecdfplot(df)
########################################################################################################################


# #################################################### MAIN ############################################################
for i in loop:
    ERB, TM = ERB_TM(N, R)
    P = matriz_dist(ERB, TM)
    S = Sombreamento_teste(P, 8)
    S2 = Sombreamento(P,8)
    D = Fast_fadding(P)
    G = Ganho(P, S, D, 19.95262315)
    G2 = Ganho(P, S2, D, 19.95262315)
    SINR_final = np.hstack((SINR_final, SINR(G,0)))
    SINR_final2 = np.hstack((SINR_final2, SINR(G2,0)))

sbn_eCDF(linear_to_dB(SINR_final))  # Curva normalizada
#sbn_eCDF(linear_to_dB(SINR_final2))  # Curva estranha

########################################################################################################################
plt.xlabel('SNR(dB)')
plt.show()
################################################## DEBUG ###############################################################
print(P, '\n\n', S, '\n\n', D, '\n\n', G, '\n\n', SINR_final)
########################################################################################################################
