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

def antenas(num, dist):  # Gerador de posicionamento de ERB's

    global sigLN
    global sigRay

    x = dist * random.rand()
    y = dist * random.rand()

    vetor1 = np.array([0,0])
    vetor2 = np.array([x,y])
    vetor3 = np.array([0])
    vetor4 = np.array([0])
    vetor5 = np.array([0])

    contador = 0
    while (contador != num):

        x = dist * random.rand()
        y = dist * random.rand()
      
        arr1 = np.array([dist* math.cos((2*math.pi/num) * contador) , dist* math.sin((2*math.pi/num) * contador)])
        arr2 = np.array([arr1[0] + x, arr1[1] + y])
        
        d = pow(arr1[0] - arr2[0],2) + pow(arr1[1] - arr2[1], 2)
        d = math.sqrt(d)

        arr3 = np.array([128.1 + 36.7 + np.log10(d)]) # dB
        arr4 = np.array([10**(lognorm(sigLN) /10)]) # dB
        arr5 = np.array([sigRay*dray(sigRay,0)])

        vetor1 = np.vstack((vetor1, arr1))
        vetor2 = np.vstack((vetor2,arr2))

        vetor3 = np.vstack((vetor3,arr3))
        vetor4 = np.vstack((vetor4, arr4))
        vetor5 = np.vstack((vetor5,arr5))

        contador += 1


    vetor4[0] = 10 **(lognorm(sigLN)/10)
    vetor5[0] = sigRay *dray(sigRay,0)

    
    return vetor1, vetor2,vetor3, vetor4, vetor5


def distancia(vet1,vet2): #  distância analítica entre dois corpos.

    d = np.array([])

    for pos in it:

        vetA = vet1[pos]
        vetB = vet2[pos]

        aux = math.sqrt(( (vetA[0] - vetB[0]) **2   + (vetA[1] - vetB[1]) **2 ))
        d = np.hstack((d,aux))

    return d    #  saida é um vetor coluna n+1 x 1 
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
    ERB, TM, P, S, D = antenas(N, R)
    
    




print(ERB, '\n\n', TM, '\n\n', P , '\n\n', S, '\n\n', D, '\n\n', '\n\n',N, '\n\n')




        


