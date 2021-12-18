import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import rad2deg, random

Ptot = np.array([0,0,0,0])
Stot = np.array([])
Dtot = np.array([])
#Counteners de Perda de potência.

GPtot = np.array([0,0,0,0])
GStot = np.array([])
GDtot = np.array([])
Gtot = np.array([0,0,0,0])
#Counteners de Ganho de Potência.

SINR =np.array([0,0,0,0])

#  Constantes
R = 500  # metros.
sigLN = 8  #  dB.
sigRay = (1/ np.sqrt(2))
PT = 13 #  43dBm (13 dB)
PN = -146 # -116 dBm (-146 dB)

def distcart(vet1, vet2):
    d = math.sqrt( (vet2[0] - vet1[1]) ** 2 + (vet2[1] - vet1[1]) ** 2 )

    return d

def lognorm(sigma, mi):
    x = random.randn()
    var = (1/sigma * math.sqrt(2 * math.pi) * abs(x)) * math.exp((-1) * ((np.log(abs(x)) - mi) ** 2)/2* (sigma **2))

    return var

def dnorm(sigma, mi):
    x = abs(random.randn())
    nor = (1/sigma* np.sqrt(2 * math.pi)) * math.exp((-1) * (1/2)* ((x - mi)/sigma) ** 2)

    return nor

ax = plt.gca()
ax.cla()
ax.set_xlim((-3*R, 3*R))
ax.set_ylim((-3*R, 3*R))

zero = np.array([0, 0])
ERB = np.array([0, 0])
ERB1 = np.array([R, 0])
ERB2 = np.array([-R*1/2, R * (math.sqrt(3)/2)])
ERB3 = np.array([-R*1/2, -R * (math.sqrt(3)/2)])

ax.plot((ERB[0]), (ERB[1]), 'o', color='b')
ax.plot((ERB1[0]), (ERB1[1]), 'o', color='b')
ax.plot((ERB2[0]), (ERB2[1]), 'o', color='b')
ax.plot((ERB3[0]), (ERB3[1]), 'o', color='b')

#  Fim da questão 01

C0 = plt.Circle((ERB[0], ERB[1]), R, color='r', fill=False)
C1 = plt.Circle((ERB1[0], ERB1[1]), R, color='r', fill=False)
C2 = plt.Circle((ERB2[0], ERB2[1]), R, color='r', fill=False)
C3 = plt.Circle((ERB3[0], ERB3[1]), R, color='r', fill=False)

ax.add_artist(C0)
ax.add_artist(C1)
ax.add_artist(C2)
ax.add_artist(C3)

t = (range(5000))

for i in t:

    r = R * math.sqrt(random.rand())
    tht = random.rand() * 2 * math.pi

    TM0 = np.array([ERB[0] + r * math.cos(tht), ERB[1] + r * math.sin(tht)])
    TM1 = np.array([ERB1[0] + r * math.cos(tht),
                   ERB1[1] + r * math.sin(tht)])
    TM2 = np.array([ERB2[0] + r * math.cos(tht),
                   ERB2[1] + r * math.sin(tht)])
    TM3 = np.array([ERB3[0] + r * math.cos(tht),
                   ERB3[1] + r * math.sin(tht)])

    ax.plot((TM0[0]), (TM0[1]), 'o', color='black')
    ax.plot((TM1[0]), (TM1[1]), 'o', color='black')
    ax.plot((TM2[0]), (TM2[1]), 'o', color='black')
    ax.plot((TM3[0]), (TM3[1]), 'o', color='black')

    #  fim da questão 02

    P0 = np.array([ 128.1 + 36.7*np.log10(distcart(TM0, ERB)), 128.1 + 36.7*np.log10(distcart(TM0, ERB1)) ,
                128.1 + 36.7*np.log10(distcart(TM0, ERB2)) ,128.1 + 36.7*np.log10(distcart(TM0, ERB3)) ])
    P1 = np.array([ 128.1 + 36.7*np.log10(distcart(TM1, ERB)), 128.1 + 36.7*np.log10(distcart(TM1, ERB1)) ,
                128.1 + 36.7*np.log10(distcart(TM1, ERB2)) ,128.1 + 36.7*np.log10(distcart(TM1, ERB3)) ])
    P2 = np.array([ 128.1 + 36.7*np.log10(distcart(TM2, ERB)), 128.1 + 36.7*np.log10(distcart(TM2, ERB1)) ,
                128.1 + 36.7*np.log10(distcart(TM2, ERB2)) ,128.1 + 36.7*np.log10(distcart(TM2, ERB3)) ])
    P3 = np.array([ 128.1 + 36.7*np.log10(distcart(TM3, ERB)), 128.1 + 36.7*np.log10(distcart(TM3, ERB1)) ,
                128.1 + 36.7*np.log10(distcart(TM3, ERB2)) ,128.1 + 36.7*np.log10(distcart(TM3, ERB3)) ])

    Ptot = np.vstack((Ptot,P0,P1,P2,P3))   # Perdas totais será organizado de forma vertical pois se trata linha-a-linha da perda de um ponto para cada antena associada.
    #  fim da questão 3a

    S0 = np.array([lognorm(sigLN,0)])
    S1 = np.array([lognorm(sigLN,0)])
    S2 = np.array([lognorm(sigLN,0)])
    S3 = np.array([lognorm(sigLN,0)])
    
    Stot = np.hstack((Stot,S0,S1,S2,S3))
    #  fim da questão 3b

    D0 = np.array([dnorm(sigRay,0) + dnorm(sigRay,0)])
    D1 = np.array([dnorm(sigRay,0) + dnorm(sigRay,0)])
    D2 = np.array([dnorm(sigRay,0) + dnorm(sigRay,0)])
    D3 = np.array([dnorm(sigRay,0) + dnorm(sigRay,0)])

    Dtot = np.hstack((Dtot,D0,D1,D2,D3))
    #  fim da questão 3c

    #  Para a questão 4, criei os vetores de potencia associado as perdas armazenadas.
    #  Sei que é um gasto desnecessário de memória, mas irei criar novos vetores para cada potência a fim de exercitar.

    GP0 = np.array([ 1/(128.1 + 36.7*np.log10(distcart(TM0, ERB))), 1/ (128.1 + 36.7*np.log10(distcart(TM0, ERB1))), 1/ (128.1 + 36.7*np.log10(distcart(TM0, ERB2))), 1/ (128.1 + 36.7*np.log10(distcart(TM0, ERB3)))])
    GP1 = np.array([ 1/ (128.1 + 36.7*np.log10(distcart(TM1, ERB))), 1/(128.1 + 36.7*np.log10(distcart(TM1, ERB1))), 1/ (128.1 + 36.7*np.log10(distcart(TM1, ERB2))), 1/ (128.1 + 36.7*np.log10(distcart(TM1, ERB3)))])
    GP2 = np.array([ 1/ (128.1 + 36.7*np.log10(distcart(TM2, ERB))), 1/ (128.1 + 36.7*np.log10(distcart(TM2, ERB1))), 1/ (128.1 + 36.7*np.log10(distcart(TM2, ERB2))), 1/ (128.1 + 36.7*np.log10(distcart(TM2, ERB3)))])
    GP3 = np.array([ 1/ (128.1 + 36.7*np.log10(distcart(TM3, ERB))), 1/ (128.1 + 36.7*np.log10(distcart(TM3, ERB1))), 1/ (128.1 + 36.7*np.log10(distcart(TM3, ERB2))), 1/ (128.1 + 36.7*np.log10(distcart(TM3, ERB3)))])

    GPtot = np.vstack((GPtot,GP0,GP1,GP2,GP3)) 

    GS0 = np.array([10 ** ((lognorm(sigLN,0) /10))])
    GS1 = np.array([10 ** ((lognorm(sigLN,0) /10))])
    GS2 = np.array([10 ** ((lognorm(sigLN,0) /10))])
    GS3 = np.array([10 ** ((lognorm(sigLN,0) /10))])

    GStot = np.hstack((GStot, GS1,GS2,GS3))

    GD0 = np.array([abs( dnorm(sigRay,0) + dnorm(sigRay,0)) ** 2])
    GD1 = np.array([abs( dnorm(sigRay,0) + dnorm(sigRay,0)) ** 2])
    GD2 = np.array([abs( dnorm(sigRay,0) + dnorm(sigRay,0)) ** 2])
    GD3 = np.array([abs( dnorm(sigRay,0) + dnorm(sigRay,0)) ** 2])

    GDtot = np.hstack((GDtot,GD0,GD1,GD2,GD3))

    GP0[0] = GP0[0] * GS0[0] * GD0[0] * PT
    GP0[1] = GP0[1] * GS0[0] * GD0[0] * PT
    GP0[2] = GP0[2] * GS0[0] * GD0[0] * PT
    GP0[3] = GP0[3] * GS0[0] * GD0[0] * PT

    GP1[0] = GP1[0] * GS0[0] * GD0[0] * PT
    GP1[1] = GP1[1] * GS0[0] * GD0[0] * PT
    GP1[2] = GP1[2] * GS0[0] * GD0[0] * PT
    GP1[3] = GP1[3] * GS0[0] * GD0[0] * PT

    GP2[0] = GP2[0] * GS0[0] * GD0[0] * PT
    GP2[1] = GP2[1] * GS0[0] * GD0[0] * PT
    GP2[2] = GP2[2] * GS0[0] * GD0[0] * PT
    GP2[3] = GP2[3] * GS0[0] * GD0[0] * PT

    GP3[0] = GP3[0] * GS0[0] * GD0[0] * PT
    GP3[1] = GP3[1] * GS0[0] * GD0[0] * PT
    GP3[2] = GP3[2] * GS0[0] * GD0[0] * PT
    GP3[3] = GP3[3] * GS0[0] * GD0[0] * PT

    Gtot = np.vstack((Gtot,GP0,GP1,GP2,GP3)) 
    #  Fim da questão 04

    SINRt = np.array([GP0[0] / (GP0[1] + GP0[2] + GP0[3] + PN),  GP1[1] / (GP1[0] + GP1[2] + GP1[3] + PN), GP2[2] / (GP2[1] + GP2[0] + GP2[3] + PN),GP3[3] / (GP3[1] + GP3[0] + GP3[2] + PN)   ])

    SINR = np.vstack((SINR, SINRt))
    #  Fim da questão 05

















print(Gtot, '\n')
print(SINR)



plt.show()
