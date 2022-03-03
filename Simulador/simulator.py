from random import Random
import numpy as np
import pandas as pd
from numpy import random
import math

#  Configuraçao inicial


# É preciso primeiramente declarar variáveis globais.

R = int(input('Insira o Raio de alcance das anteas: '))
N = int(input('Insira o numero de anteas desejado: '))
loop = int(input('Insira o numero de loops que deve acontecer na simulação: '))
loop = range(loop)

PT = 19.95262315 #  43 dBm
PN = (2.5118864315 * 10 **(-15))
sigLN = 8
sigRay = (1/ np.sqrt(2))

#  métodos

def antenas(num, dist):  # Gerador de posicionamento de ERB's

    vetor = np.array([0,0])
    contador = 0
    while (contador != num):
        
        arr = np.array([dist* math.cos((2*math.pi/num) * contador) , dist* math.sin((2*math.pi/num) * contador)])
        vetor = np.vstack((vetor, arr))
        contador += 1

    return vetor

def distancia(vet1,vet2): #  distância analítica entre dois corpos.

    d = np.array([])

    i = np.size(vet1) / 2

    for pos in range(int(i)):

        vetA = vet1[pos]
        vetB = vet2[pos]

        aux = math.sqrt(( (vetA[0] - vetB[0]) **2   + (vetA[1] - vetB[1]) **2 ))
        d = np.hstack((d,aux))

    return d    

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

def TMn(vet1, const1, const2): #  Função geradora do posicionamento dos móveis.

    TM = np.array([0,0])

    for vet in vet1:
        aux = np.array([vet[0] + (const1*const2), vet[1] + (const2*const1)])
        TM = np.vstack((TM, aux))

    TM = TM[1:len(vet1) -1]

    return TM

def Pn(vet1, vet2):  # Função geradora do ganho em W da fonte. (Ganho linear)

    P = np.array([0])


    for i in range(len(vet1)):

        d = distancia(vet1,vet2)
        P = np.vstack((P, [128.1 + 36.7 + np.log10(d[i])])) #  P gerado primeiramente em dB.

    for i in P:

        i = 10 ** (i/10)  #  e então convertido para escala linear.

    P = P[1:len(P) -1]

    return P

def Sombreamento(vet1,vet2,sd): #  Função geradora para valores de sombreamento:

    d = np.array([])
    S = np.array([0])


    for i in vet1:
        for j in vet2:

            aux = math.sqrt( (i[0] - j[0])**2 + (i[1] - j[1]) **2 )
            d = np.hstack((d, aux))

            if aux <= R:

                S = np.vstack((S,lognorm(sd)))

            else:

                S = np.vstack((S, 0))


    return S
            
def Desvanecimento(): #  Função geradora para valores de Desvanecimento rápido.



#  main

ERB = antenas(N, R)

#  Simulação:  

for i in loop:

    #  Atualizar estado do sistema:

    #  definindo variáveis aleatórias de posicionamento de TM.
    x = R* math.sqrt(random.rand())
    tht = random.rand() *2 *( math.pi)

    #  definindo variáveis aleatórias de posicionamento de TM.

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
    TM = TMn(ERB, x, tht)  #  Matriz 2 x n+1
    
    P = Pn(TM,ERB)  # Vetor coluna n + 1 x 1   
    S = Sombreamento(ERB, TM,sigLN) # Vetor coluna (n+1)^2 x 1 
    D = 
    





        


