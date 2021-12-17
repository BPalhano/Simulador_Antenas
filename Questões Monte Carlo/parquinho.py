import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import rad2deg, random

R = 500  # metros.
T = np.array([0, 2*math.pi])

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

t = (range(2))

for i in t:

    r = R * math.sqrt(random.rand())
    tht = random.rand() * 2 * math.pi
    r1 = R * math.sqrt(random.rand())
    tht1 = random.rand() * 2 * math.pi
    r2 = R * math.sqrt(random.rand())
    tht2 = random.rand() * 2 * math.pi
    r3 = R * math.sqrt(random.rand())
    tht3 = random.rand() * 2 * math.pi

    TM0 = np.array([ERB[0] + r * math.cos(tht), ERB[1] + r * math.sin(tht)])
    TM1 = np.array([ERB1[0] + r1 * math.cos(tht1),
                   ERB1[1] + r1 * math.sin(tht1)])
    TM2 = np.array([ERB2[0] + r2 * math.cos(tht2),
                   ERB2[1] + r2 * math.sin(tht2)])
    TM3 = np.array([ERB3[0] + r3 * math.cos(tht3),
                   ERB3[1] + r3 * math.sin(tht3)])

    ax.plot((TM0[0]), (TM0[1]), 'o', color='black')
    ax.plot((TM1[0]), (TM1[1]), 'o', color='black')
    ax.plot((TM2[0]), (TM2[1]), 'o', color='black')
    ax.plot((TM3[0]), (TM3[1]), 'o', color='black')

#  fim da questão 02


plt.show()
