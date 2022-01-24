import math
import random

import numpy as np

class CartPole():

    def __init__(self):
        self.length = 0.5
        self.poleMass = 0.1
        self.cartMass = 1
        self.gravity = 9.8
        #self.poleAngle = 0
        self.firstDeriv = 0
        self.secondDeriv = 0
        self.cartLoc = 0
        self.cartAcc = 0                # Cart acceleration
        self.BB = 0                     # Bang bang force
        self.maxMag = 0.21              # Maximum angle before it fails
        self.minPos = -2.4
        self.maxPos = 2.4
        self.timeStep = 0.02
        self.episodeLength = 300

        self.poleAngle = random.uniform(-self.maxMag, self.maxMag)
        self.firstDeriv = 0

    def step(self, action):
        # Her kommer algoritmen fra prosjekt dokumentet
        pass
