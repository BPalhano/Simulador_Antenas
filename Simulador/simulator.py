################################################ BIBLIOTECAS ###########################################################
import numpy as np
import pandas as pd
import seaborn as sbn
from scipy import stats as scs
from matplotlib import pyplot as plt
from numpy import random
########################################################################################################################

############################################  VARIÁVEIS GLOBAIS ########################################################
N = int(input('Numero de usuários: '))
R = 200
loop = int(input('Quantos laços de repetição: '))

print('\n\n')

c = 299792458
freq = 6000000000 # 6 * 10^{9}. Hz
_lambda = c / freq
loop = range(loop)
it = range(N + 1)
SINR_final = np.array([])
SINR_final2 = np.array([])
SINR_final3 = np.array([])


########################################################################################################################

# #################################### CONVERSOR DE UNIDADES ###########################################################


def dB_to_linear(x):  # conversor de dB para escala linear

    return 10 ** (x / 10)  # retorno em escala linear


def linear_to_dB(x):  # conversor de escala linear para dB

    return 10 * np.log10(x) # retorno em dB


def dBm_to_linear(x):  # conversor de dBm para escala linear

    return 10**(x/10 - 3) # retorna em escala linear

########################################################################################################################


# #################################### GERADORES DE POSICIONAMENTO #####################################################
def ERB_TM(num, dist):
    ERB = np.array([0, 0, 100])
    ap = dist * np.sqrt(3)/2
    TM = np.array([R * random.uniform(0, 1), R * random.uniform(0, 1), 0])

    contador_ERB = 0

    while contador_ERB != 6:
        arr1 = np.array(
            [2*ap * np.cos((np.pi / 6) + contador_ERB*np.pi/3), 2*ap * np.sin((np.pi / 6) + contador_ERB*np.pi/3), 100])
        ERB = np.vstack((ERB, arr1))

        arr2 = np.array(
            [arr1[0] + dist * random.uniform(0, 1), arr1[1] + dist * random.uniform(0, 1), 100 * random.uniform(-1, 0)])
        TM = np.vstack((TM, arr2))

        contador_ERB += 1

    if N > contador_ERB:
        contador_TM = 0

        while contador_TM != num:
            aux = np.array([6 * dist * random.normal(0, 1), 6 * dist * np.sin(np.pi / 3) * random.uniform(0, 1),
                            100 * random.uniform(0, 1)], dtype=object)

            TM = np.vstack((TM, aux))
            contador_TM += 1

    return ERB, TM


def matriz_dist(vet1, vet2):  # matriz de distâncias de ERB i para TM j

    d = np.array([])

    for i in vet1:

        for j in vet2:
            dist = np.sqrt(pow((i[0] - j[0]), 2) + pow((i[1] - j[1]), 2) + pow((i[2] - j[2]), 2))
            d = np.hstack((d, dist))

    return d  # retorna uma matriz i x j
########################################################################################################################


# ################################# GERADORES DE RUIDO #################################################################


def Path_loss(vet1):  # Gerador de interferência linear.

    vet1 = vet1.copy()

    for i in range(len(vet1)):
        vet1[i] = dBm_to_linear(128.1 + 3.67 * np.log10(vet1[i]))

    return vet1  # em W

def Sombreamento_teste(vet1, sd):
    aux = vet1.copy()

    rv = random.lognormal(mean=0, sigma=sd)

    while rv > 100:
        rv = random.lognormal(mean=0, sigma=sd)

    for i in range(len(vet1)):
        aux[i] = dBm_to_linear(rv)

    return aux  # em W


def Sombreamento(vet1, sd,xdim=1, ydim=1):

    aux = vet1.copy()

    for i in range(len(vet1)):
        rv = sd*random.randn(xdim,ydim)

        aux[i] = dBm_to_linear(rv)
    return aux #  em W


def Fast_fadding(vet1, xdim=1, ydim=1): # Rice
    aux = vet1.copy()
    rv = np.array([])

    for i in range(len(vet1)):
        for k in range (10):
            rv = np.hstack ((rv ,abs((1 / (2 ** 0.5)) * scs.rice.rvs ( xdim , ydim ) + (1 / (2 ** 0.5)) *
                                                                                    scs.rice.rvs ( xdim ,ydim ) * 1j)**2))
        aux[i] = dBm_to_linear(np.mean(rv))

    return aux  # em W

def Fast_fadding2(vet1, xdim=1, ydim=1): #  Nakagami
    aux = vet1.copy()
    rv = np.array([])

    for i in range(len(vet1)):
        for k in range(10):
            rv = np.hstack((rv, abs((1 / (2 ** 0.5)) * scs.nakagami.rvs(5,xdim, ydim) + (1 / (2 ** 0.5)) *
                                                                                scs.nakagami.rvs(5,xdim,ydim)*1j)**2 ))

        aux[i] = dB_to_linear(np.mean(rv))

    return aux # em w


def Ganho(vet1, vet2, vet3, K):  # Ganho total do sistema

    G = vet1.copy()

    output = np.multiply(vet1,vet2)
    output = np.multiply(output,vet3)
    output = np.multiply(output,K)

    return output  # em W


def SINR(vet1, const):
    matrix = vet1.copy()
    trace = 0

    for i in range(0, 7):
        trace += matrix[7 * i]

    tot = np.sum(matrix)

    cnt = tot - trace

    SNR = np.array([])

    for i in range(7):
        val = matrix[i * 7] / (const + cnt)

        SNR = np.hstack((SNR, val))

    return SNR

def SINR_test(vet1,const):

    matrix = np.array([])
    aux = np.array([])

    for i in range(N):

        for j in range(7):
            aux = np.hstack((aux, vet1[j]))

        if i == 0:
            matrix = np.hstack((matrix, aux))
            aux = np.array([])

        else:
            matrix = np.vstack((matrix, aux))
            aux = np.array([])

    output = np.array([])

    for i in range(7):

        x = (matrix[i][i]/ ((np.sum(matrix) - np.trace(matrix)) - const))
        output = np.hstack((output,x))

    return x


def sbn_eCDF(vet1):
    dataset = vet1.copy()
    df = pd.DataFrame(dataset)

    sbn.ecdfplot(df)
########################################################################################################################


# #################################################### MAIN ############################################################
for i in loop:

    ERB, TM = ERB_TM(N, R)
    P = Path_loss(matriz_dist(ERB, TM))
    S = Sombreamento_teste(P, 8) # 1 valor p/ cluster.
    S2 = Sombreamento(P,8) # 1 valor por usuário, não interpolado.
    D = Fast_fadding2(P)
    G = Ganho(P, S, D, dBm_to_linear(43))
    G2 = Ganho(P, S2, D, dBm_to_linear(43))
    SINR_final = np.hstack((SINR_final, SINR_test(G, dBm_to_linear(-116))))
    SINR_final2 = np.hstack((SINR_final2, SINR(G2, dBm_to_linear(-114))))

#PLOT DE SINR:
#sbn_eCDF(linear_to_dB(SINR_final))  # Curva 1,
sbn_eCDF(linear_to_dB(SINR_final2))  # Curva 2,
#sbn_eCDF(linear_to_dB(SINR_final3))  # Curva 3

#PLOT DE SNR:
#sbn_eCDF(linear_to_dB(G))
#sbn_eCDF(linear_to_dB(G2))

########################################################################################################################
plt.show()
################################################## DEBUG ###############################################################
print('Valores de SINR:','\n\n', linear_to_dB(SINR_final2), '\n\n', linear_to_dB(SINR_final), '\n\n')
print('Valores de SNR:','\n\n', linear_to_dB(G), '\n\n', linear_to_dB(G2))
########################################################################################################################
