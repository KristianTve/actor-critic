import math
import random

import numpy as np

class CartPole():

    def __init__(self):
        self.length = 0.5
        self.poleMass = 0.1
        self.cartMass = 1
        self.g = 9.81                    # Gravity

        self.firstDeriv = 0             # First temporal derivative of pole angle
        self.secondDeriv = 0            # Second temporal derivative of pole angle
        self.cartAcc = 0                # Cart acceleration
        self.BB = 0                     # Bang bang force
        self.maxMag = 0.21              # Maximum angle before it fails
        self.minPos = -2.4
        self.maxPos = 2.4
        self.timeStep = 0.02
        self.episodeLength = 300

        self.poleAngle = random.uniform(-self.maxMag, self.maxMag)  # Pole angle (theta)
        self.firstDeriv = 0

    def step(self, action):
        # Her kommer algoritmen fra prosjekt dokumentet

        self.secondDeriv = (self.g * np.sin(self.poleAngle) +
                            (np.cos(self.poleAngle)*(-self.BB-self.poleMass*self.length*self.secondDeriv*np.sin(self.poleAngle)))
                            /(self.poleMass + self.cartMass))\
                           /(self.length*(4/3 - (self.poleMass*np.cos(self.poleAngle)**2)/(self.poleMass + self.cartMass)  ))

        self.cartAcc = (self.BB + self.poleMass*self.length*(self.firstDeriv**2*np.sin(self.poleAngle) - self.secondDeriv(self.poleAngle) * np.cos(self.poleAngle)))\
                       /(self.poleMass + self.cartMass)



