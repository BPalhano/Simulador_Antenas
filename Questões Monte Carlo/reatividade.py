#  bibliotecas 

import numpy as np
import math
from numpy import random
from numpy.core.shape_base import vstack

# constantes 

loop = range(5000)
R = 500
x = math.cos((2*math.pi)/3)
y = math.sin((2*math.pi)/3)
sigLN = 8
sigRay = (1/ np.sqrt(2))
PT = 19.95262315 #  43 dBm
PN = (2.5118864315 * 10 **(-15))

ERB1 = np.array([0,0])
ERB2 = np.array([R,0])
ERB3 = np.array([R * x , R * y ])
ERB4 = np.array([-R * x , R * y ])

#  Vetores de resultados

Ptot = np.array([0,0,0,0]) #  Perda de potencia por distancia
Stot = np.array([0,0,0,0])  # Sombreamento
Dtot = np.array([0,0,0,0])  #  Desvanecimento Rapido

#  Vetores de perda de potencia.

GPtot = np.array([0,0,0,0])  # ganho por perda de potencia por distancia
GStot = np.array([0,0,0,0])  # ganho por sombreamento 
GDtot = np.array([0,0,0,0])  #  Ganho por desvanecimento rapido
Gtot = np.array([0,0,0,0])  # Ganho total

#  Vetores de Ganho de Potência.

SINR = np.array([0,0,0,0])

# funçoes:

def distcart(vet1, vet2):
    d = math.sqrt( (vet2[0] - vet1[1]) ** 2 + (vet2[1] - vet1[1]) ** 2 )

    return d

def lognorm(sigma):

    x = random.rand() * sigma

    return x

def dray(sigma, mi):
   
    x = random.normal(loc=mi, scale=sigma)
    y = random.normal(loc=mi, scale=sigma)

    h = abs(x**2 - y**2) #  |H(i,j)| ^2

    return abs(h)

def media(vet1):

    mi = 0
    for i in vet1:
        for j in i:

            mi += j
    
    mi /= (len(loop) * 4)

    return mi

def sig(vet1):

    sigma = math.sqrt( media((vet1) ** 2) - (media(vet1))**2 )

    return sigma

def TMn(vet1, const1, const2):

    TM = np.array([vet1[0] + (const1*const2), vet1[1] + (const1*const2)])

    return TM

def Pn(vet1, vet2,vet3, vet4 , vet5):

    P = ([ 128.1 + 36.7 * np.log10(distcart(vet1, vet2)) , 128.1 + 36.7 * np.log10(distcart(vet1, vet3)) ,128.1 + 
            36.7 * np.log10(distcart(vet1, vet4)) , 128.1 + 36.7 * np.log10(distcart(vet1, vet5)) ])

    return P

def Gn(TM, vet1,vet2,vet3, const):

    GT = np.array([const * vet1[TM+1][0]* vet2[TM+1][0]* vet3[TM+1][0],
                    const * vet1[TM+1][1]* vet2[TM+1][1]* vet3[TM+1][1],
                    const * vet1[TM+1][2]* vet2[TM+1][2]* vet3[TM+1][2],
                    const * vet1[TM+1][3]* vet2[TM+1][3] *vet3[TM+1][3]])

    return GT

def CDF_Gaussiana(mean, sig, vet):
     
    for x in vet:
        for y in x:

            y = (1/2) * (1 + math.erf( (y - mean)/ (sig * math.sqrt(2))))

    return vet
#  main

for i in loop:

    GPloc = np.array([0,0,0,0])
    GSloc = np.array([0,0,0,0])
    GDloc = np.array([0,0,0,0])


    r = R*math.sqrt(random.rand())
    tht = random.rand() * 2 * math.pi

    TM1 = TMn(ERB1, r,tht)
    TM2 = TMn(ERB2, r,tht)
    TM3 = TMn(ERB3, r,tht)
    TM4 = TMn(ERB4, r,tht)

    P1 = Pn(TM1,ERB1,ERB2,ERB3,ERB4)
    P2 = Pn(TM2,ERB1,ERB2,ERB3,ERB4)
    P3 = Pn(TM3,ERB1,ERB2,ERB3,ERB4)
    P4 = Pn(TM4,ERB1,ERB2,ERB3,ERB4)

    Ptot = np.vstack((Ptot, P1,P2,P3,P4))

    S1 = np.array([lognorm(sigLN), lognorm(sigLN),lognorm(sigLN),lognorm(sigLN)])
    S2 = np.array([lognorm(sigLN), lognorm(sigLN),lognorm(sigLN),lognorm(sigLN)])
    S3 = np.array([lognorm(sigLN), lognorm(sigLN),lognorm(sigLN),lognorm(sigLN)])
    S4 = np.array([lognorm(sigLN), lognorm(sigLN),lognorm(sigLN),lognorm(sigLN)])

    Stot = np.vstack((Stot, S1,S2,S3,S4))

    D1 = np.array([sigRay * dray(sigRay,0), sigRay * dray(sigRay,0), sigRay * dray(sigRay,0), sigRay * dray(sigRay,0)])
    D2 = np.array([sigRay * dray(sigRay,0), sigRay * dray(sigRay,0), sigRay * dray(sigRay,0), sigRay * dray(sigRay,0)])
    D3 = np.array([sigRay * dray(sigRay,0), sigRay * dray(sigRay,0), sigRay * dray(sigRay,0), sigRay * dray(sigRay,0)])
    D4 = np.array([sigRay * dray(sigRay,0), sigRay * dray(sigRay,0), sigRay * dray(sigRay,0), sigRay * dray(sigRay,0)])

    Dtot = np.vstack((Dtot,D1,D2,D3,D4))

    #  Ganho de potencias
    GP1 = np.array([1/P1[0], 1/P1[1], 1/P1[2], 1/ P1[3]])
    GP2 = np.array([1/P2[0], 1/P2[1], 1/P2[2], 1/ P2[3]])
    GP3 = np.array([1/P3[0], 1/P3[1], 1/P3[2], 1/ P3[3]])
    GP4 = np.array([1/P4[0], 1/P4[1], 1/P4[2], 1/ P4[3]])
    GPloc = np.vstack((GPloc,GP1,GP2,GP3,GP4))

    GPtot = vstack((GPtot,GPloc))

    GS1 = np.array([ 10**(S1[0]/10), 10**(S1[1]/10), 10**(S1[2]/10), 10**(S1[3]/10)])
    GS2 = np.array([ 10**(S2[0]/10), 10**(S2[1]/10), 10**(S2[2]/10), 10**(S2[3]/10)])
    GS3 = np.array([ 10**(S3[0]/10), 10**(S3[1]/10), 10**(S3[2]/10), 10**(S3[3]/10)])
    GS4 = np.array([ 10**(S4[0]/10), 10**(S4[1]/10), 10**(S4[2]/10), 10**(S4[3]/10)])
    GSloc = np.vstack((GSloc, GS1,GS2,GS3,GS4))

    GStot = vstack((GStot,GSloc))

    GD1 = np.array([(D1[0]),(D1[1]),(D1[2]),(D1[3])])
    GD2 = np.array([(D2[0]),(D2[1]),(D2[2]),(D2[3])])
    GD3 = np.array([(D3[0]),(D3[1]),(D3[2]),(D3[3])])
    GD4 = np.array([(D4[0]),(D4[1]),(D4[2]),(D4[3])])
    GDloc = np.vstack((GDloc,GD1,GD2,GD3,GD4))
    
    GDtot = vstack((GDtot,GDloc))

    G1 = np.array([Gn(0,GPloc, GSloc, GDloc,42)])
    G2 = np.array([Gn(1,GPloc, GSloc, GDloc,42)])
    G3 = np.array([Gn(2,GPloc, GSloc, GDloc,42)])
    G4 = np.array([Gn(3,GPloc, GSloc, GDloc,42)])
    Gloc = np.vstack((G1,G2,G3,G4))

    Gtot = vstack((Gtot, Gloc))

    soma = Gloc[0][1]+Gloc[0][2]+ Gloc[0][3] + Gloc[1][0]+Gloc[1][2]+ Gloc[1][3] + Gloc[2][2]+Gloc[2][1]+ Gloc[2][3] + Gloc[3][0]+Gloc[3][1]+ Gloc[3][2] 
    aux = np.array([ (Gloc[0][0]/(PN +soma)) , (Gloc[1][1]/(PN +soma)) , (Gloc[2][2]/(PN +soma)), (Gloc[3][3]/(PN +soma)) ])

    SINR = np.vstack((SINR, abs(aux)))




#  todos os vetores iniciados com [0,0,0,0] serao cortados para retirar essa "linha morta"
Ptot = (Ptot[1:4 * len(loop) + 1, 0:5])
Stot = (Stot[1:4 * len(loop) + 1, 0:5])
Dtot = (Dtot[1:4 * len(loop) + 1, 0:5])

GPtot = (GPtot[1:4 * len(loop) + 1, 0:5])
GStot = (GStot[2:4 * len(loop) + 1, 0:5])
GDtot = (GDtot[1:4 * len(loop) + 1, 0:5])
Gtot = (Gtot[1:4 * len(loop) + 1, 0:5])
SINR = (SINR[1:4 * len(loop) + 1, 0:5])

print('Com base na Lei dos Grandes Numeros (LGN), a CDF dos valores de SINR sera a CDF de uma distribuiçao gaussiana.' )

print('A media de valores relativos a SINR: ',media(SINR), '\n')
print('O desvio padrao dos valores de SINR: ',sig(SINR), '\n')
print('Vetor de CDF: ','\n', CDF_Gaussiana(media(SINR),sig(SINR), SINR))

