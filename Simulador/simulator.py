from random import Random
import numpy as np
import pandas as pd
from numpy import random
import math

#  Configuraçao inicial


# É preciso primeiramente declarar variáveis globais.

R = int(input('Insira o Raio de alcance das antenas: '))
N = int(input('Insira o número de antenas desejado: '))
loop = int(input('Insira o número de loops que deve acontecer na simulação: '))
loop = range(loop)
it = range(N+1)

PT = 19.95262315 #  43 dBm
PN = (2.5118864315 * 10 **(-15))
sigLN = 8
sigRay = (1/ np.sqrt(2))



#  métodos

def antenas(num, dist):  # Gerador de posicionamento de ERB's

    x = dist * random.rand()
    y = dist * random.rand()

    vetor1 = np.array([0,0])
    vetor2 = np.array([x,y])
  
    contador = 0
    while (contador != num):

        x = dist * random.rand()
        y = dist * random.rand()
      
        arr1 = np.array([dist* math.cos((2*math.pi/num) * contador) , dist* math.sin((2*math.pi/num) * contador)])
        arr2 = np.array([arr1[0] + x, arr1[1] + y])
        
        vetor1 = np.vstack((vetor1, arr1))
        vetor2 = np.vstack((vetor2,arr2))
        
        contador += 1

    
    return vetor1, vetor2


def distancia(vet1,vet2): #  distância analítica entre dois corpos.

    d = np.array([])

    for pos in it:

        vetA = vet1[pos]
        vetB = vet2[pos]

        aux = math.sqrt(( (vetA[0] - vetB[0]) **2   + (vetA[1] - vetB[1]) **2 ))
        d = np.hstack((d,aux))

    return d    #  saida é um vetor coluna n+1 x 1 

def lognorm(sigma):  #  Gerador de números log-normais.

    x = random.rand() * sigma

    return x

def dray(sigma, mi):  #  Gerador de valores em distribuição de Rayleigh
   
    x = random.normal(loc=mi, scale=sigma)
    y = random.normal(loc=mi, scale=sigma)

    h = abs(x**2 - y**2) #  |H(i,j)| ^2

    return abs(h)

def media(vet1):  # media de valores em um vetor qualquer.

    mi = 0
    for i in vet1:
        for j in i:

            mi += j
    
    mi /= ((np.size(vet1)))

    return mi

def sig(vet1):  #  desvio padrão de valores em um vetor qualquer.

    sigma = math.sqrt( media((vet1) ** 2) - (media(vet1))**2 )

    return sigma

def Pn(vet1, vet2):  # Função geradora do ganho em W da fonte. (Ganho linear)

    P = np.array([0])
    d = distancia(vet1,vet2)

    for i in range(len(vet1)):

        P = np.vstack((P, [128.1 + 36.7 + np.log10(d[i])])) #  P gerado primeiramente em dB.

    for i in P:

        i = 10 ** (i/10)  #  e então convertido para escala linear.

    P = P[1:len(P) -1]

    return P

def Sombreamento(vet1,sd): #  Função geradora para valores de sombreamento em W:

    S = np.array([0])

    for element in vet1:

        x = 10**(lognorm(sd) /10)
        S = np.vstack((S, x))

    S = S[1: len(vet1) -1]
    return S
            
def Desvanecimento(vet1, sd): #  Função geradora para valores de Desvanecimento rápido em W. 

    D = np.array([0])

    for element in vet1:

        x = sd*dray(sd,0)
        D = np.vstack((D, x))
    
    D = D[1: len(vet1) - 1]

    return D



#  main

#  Simulação:  

for i in loop:

    #  Atualizar estado do sistema:

    #  Tornando os conteiners utilizados nulos para o procedimento matemático.
    TM = np.array([])
    P = np.array([])
    S = np.array([])
    D = np.array([])
    
    GP = np.array([])
    GS = np.array([])
    GD = np.array([])
    
    G = np.array([])

    #  Mudança de valores devido a atualização do estado do sistema.
    ERB, TM = antenas(N, R)
    
    P = Pn(TM,ERB)  # Vetor coluna n + 1 x 1   em W
    S = Sombreamento(ERB,sigLN) # Vetor coluna n+1 x 1 com todos os valores para ERBi TMi  em dB
    D = Desvanecimento(ERB, sigRay)  # vetor coluna n+1 x 1 com tods os Fast fadding para ERBi e TMi  em dB
    




print(ERB, '\n\n', TM, '\n\n', P , '\n\n', S, '\n\n', D, '\n\n', '\n\n',N)




        


