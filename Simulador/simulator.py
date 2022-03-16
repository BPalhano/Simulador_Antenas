import numpy as np
import pandas as pd
from numpy import random
import math
import matplotlib.pyplot as plt

#  Configuraçao inicial


# É preciso primeiramente declarar variáveis globais.

R = int(input('Insira o Raio de alcance das antenas: '))
N = int(input('Insira o número de antenas desejado: '))
loop = int(input('Insira o número de loops que deve acontecer na simulação: '))

print('\n\n')

loop = range(loop)
it = range(N+1)

PT = 19.95262315 #  43 dBm
PN = (2.5118864315 * 10 **(-15))
sigLN = 8
sigRay = (1/ np.sqrt(2))

SINR_final = np.array([])

def antenas(num, dist):  # Gerador de posicionamento de ERB's

    
    x = R*math.sqrt(random.rand()) * random.rand() * 2 * math.pi
    y = R*math.sqrt(random.rand()) * random.rand() * 2 * math.pi
    

    vetor1 = np.array([0,0])
    vetor2 = np.array([x,y])

    contador = 0
    
    while (contador != num):

        x = R*math.sqrt(random.rand()) * random.rand() * 2 * math.pi
        y = R*math.sqrt(random.rand()) * random.rand() * 2 * math.pi

        arr1 = np.array([dist* math.cos((2*math.pi/num) * contador) , dist* math.sin((2*math.pi/num) * contador)])
        arr2 = np.array([arr1[0] + x, arr1[1] + y])
        
        vetor1 = np.vstack((vetor1, arr1))
        vetor2 = np.vstack((vetor2,arr2))




        contador += 1

    
    return vetor1, vetor2

def lognorm(sigma):  #  Gerador de números log-normais.

    x = random.rand() * sigma

    return x

def dray(sigma, mi):  #  Gerador de valores em distribuição de Rayleigh
   
    x = random.normal(loc=mi, scale=sigma)
    y = random.normal(loc=mi, scale=sigma)

    h = abs(x**2 - y**2) #  |H(i,j)| ^2

    return abs(h)

def matriz_dist(vet1, vet2):  #  matriz de distâncias de ERB i para TM j

    d = np.array([])

    for i in vet1:
        
      for j in vet2:

            dist = math.sqrt( pow( (i[0] - j[0]), 2 ) + pow((i[1] - j[1]), 2 ) )
            d = np.hstack((d, dist))
  
    return d  # retorna uma matriz i x j

def dB_to_linear(x):  # conversor de dB para escala linear

    x = 10 **(x/10)

    return x #  retorno em escala linear

def linear_to_dB(x):  # conversor de escala linear para dB

    x = 10* np.log10(x)

    return x #  retorno em dB

def Pot_linear(vet1):  #  Gerador de interferência linear.

    vet1 = vet1.copy()

    for i in range(len(vet1)):

        vet1[i] = dB_to_linear(164.8 + np.log10(vet1[i]))

    return vet1  #  em W

def Sombreamento(vet1,sd):  #  Gerador de Sombreamento.

    aux = vet1.copy()

    for i in range(len(vet1)):

        aux[i] = dB_to_linear(lognorm(sd))

    return aux  # em W

def Fast_fadding(vet1, sd):  #  Gerador de Desvanecimento Rápido

    aux = vet1.copy()

    for i in range(len(vet1)):

        aux[i] = dB_to_linear(sd*dray(sd,0))

    return aux # em W

def Ganho(vet1,vet2,vet3, K):  #  Ganho total do sistema

    G = vet1.copy()

    for i in range(len(vet1)):

        G[i] = vet1[i] * vet2[i] * vet3[i] * K

        

    return G  # em W

def Soma(vet1):  #  Função para somar todas os elementos de um vetor 

    s = 0

    for i in vet1:
        s += i
    
    return s

def SINR(num, vet1, K):  #  Função para calcular o SINR

    matrix = vet1.copy()
    trace = np.array([])

    contador = 0
    for element in range(len(matrix)):

        if (contador % (num+1) == 0):

            trace = np.hstack((trace, matrix[element]))
        contador += 1

    Sum_trace = Soma(trace)
    Sum_matrix = Soma(matrix)

    Sum_const = Sum_matrix - Sum_trace

    SNR = np.array([])

    for element in range(len(trace)):

        aux = (trace[element] / (Sum_const + K))

        SNR = np.hstack((SNR,aux))


        




    return SNR

def eCDF(vet1):  #  Função para plotar a CDF empírica do sistema.

    dataset = vet1.copy()
    df = pd.DataFrame(dataset.data, columns= dataset.feature_names)

    x = np.sort(df)
    y = np.arange(1, len(df) +1) / len(df)
    
    plt.plot(x, y, marker = '.', linestyle = 'none')
    plt.margins(0.1)

    plt.show()


#  main

#  Simulação:

for i in loop:#  Simulação

    # Atualização do Estado 
    ERB, TM = antenas(N, R)  #  Armazenamento da posição do móveis e antenas.
    
    
    P =  Pot_linear(matriz_dist(ERB, TM))  #  Armazenamento do Ruído Linear do estado atual 
    S = Sombreamento(P, sigLN)    #  Armazenamento do Ruído de Sombreamento do estado atual.
    D = Fast_fadding(P, sigRay)  #  Armazenamento do Ruído de Desvanecimento Rápido do estado atual.

    G = Ganho(P,S,D, PT)  #  Armazenamento do Ganho total em W do estado atual.
    
    #  Organização e Armazenamento Parcial (SINR)
    if i == 0:
        SINR_final = np.hstack( (SINR_final,(SINR(N,G, PN)) ))
    
    else:
        SINR_final = np.vstack( (SINR_final, (SINR(N,G,PN)) ))  
    #  Parar Simulação?

#  Organização e Armazenamento final
SINR_final = linear_to_dB(SINR_final)


eCDF(SINR_final)
#  fim
