import numpy as np 
#import matplotlib.pyplot as plt

"""

- Ganho da fonte
- Perda ambiental
- perda de propagaçao
- efeitos entropicos

"""
#  Definindo parametros basicos:

c = 300000000 # m/s.
freq = float(input('Frequencia(Hz): ')) #  MHz.
lamb = c/ freq #  m.

Xa = float(input('Posiçao da antena no eixo x(km): ')) #  Km.
Ya = float(input('Posiçao da antena no eixo y(km): ')) #  Km.
Za = float(input('Posiçao da antena no eixo z(km): ')) #  Km.

PosAnt = np.array([Xa, Ya, Za])

Xm = float(input('Posiçao do mobile no eixo x(km): ')) #  Km.
Ym = float(input('Posiçao do mobile no eixo y(km): ')) #  Km.
Zm = float(input('Posiçao do mobile no eixo z(km): ')) #  Km.

PosMob = np.array([Xm, Ym, Zm])

DisAM = np.sqrt(pow((PosAnt[0] - PosMob[0]),2) +  pow((PosAnt[1] - PosMob[1]),2) + pow(PosAnt[2] - PosMob[2],2))  #  Km

#  Modular para Earth-Plane Loss

Lep = 40*np.log10(DisAM) -20*np.log10(PosMob[2]) - 20*np.log10(PosAnt[2])  #  dB.

#inserindo um clutter factor 20 dB

K = 20

#Perda de sinal com CLutter 

Lepc = 40 * np.log10(DisAM) - 20 * np.log10(PosMob[2]) - 20* np.log10(PosAnt[2]) + K  #  dB.

#  Modular para Free-Space Loss


Lfs = 32.4 + np.log10(DisAM) + 20 * np.log10(freq)  #  dB.


#  Perda de sinal empirica (Okumura-Hata) (200 MHz to 2 GHz)

Loh = 69.55 +26.16 * np.log10(freq) - 13.82 * np.log10(PosAnt[2]) - (44.9 -6.55 * np.log10(PosAnt[2]))* np.log10(DisAM)

print('Insira um numero listado: ')
arg = input('1.[CIDADE URBANIZADA], 2.[CIDADE SUBURBANIZADA], 3.[RURAL]:')

if arg == 1:

    if freq >= 300:

        Loh = Loh - 3.2* pow(( np.log10(11.75*PosMob[2])),2)  #  dB.


    else:
    
        Loh = Loh - 8.29 * pow(( np.log10(1.54*PosMob[2])),2)  #  dB.



elif arg == 2:

    Loh = Loh - 5.4 + 2* pow( (np.log10(freq / 28)) ,2)  #  dB.


else:

    Loh = Loh - 40.94 + 4.78*pow((np.log10(freq)) ,2) - 18.33 * np.log10(freq)  #  dB.


# modelo COST231 - Hata (1500 MHz < fc < 2000 MHz) 

Lch = 46.3 + 33.9 * np.log10(freq) - 13.82 * np.log10(PosMob[2]) + (44.9 -6.55 * np.log10(PosAnt[2]))* np.log10(DisAM)  #  dB.

    
if arg == 1:

    Lch = Lch - 3.2* pow(( np.log10(11.75*PosMob[2])),2) + 3   #  dB.


else:

    Lch = Lch - 3.2* pow(( np.log10(11.75*PosMob[2])),2) + 0  #  dB.

# Modelo de Lee 

Lml = 29 - 20* np.log10(PosAnt[2] - PosMob[2]) - 10 * np.log10(PosAnt[2])

if arg == 1:

    n = 3.98
    Po = -73.75 
    #Utilizei para ambos um valor medio tabelado no livro de Antenas do Saunders.

    Lml = Lml - Po + 10*n* np.log10(DisAM)

elif arg == 2:

    n = 3.84
    Po = -61.7

    Lml = Lml - Po + 10*n* np.log10(DisAM)

else:

    n = 4.35
    Po = -49

    Lml = Lml - Po + 10*n* np.log10(DisAM)

#  Modelo de Ibrahim e Parsons.

Lip = 40* np.log10