import math
import random

import numpy as np

class CartPole:

    def __init__(self):
        self.length = 0.5
        self.poleMass = 0.1
        self.cartMass = 1
        self.g = 9.81                    # Gravity

        self.poleSpeed = 0              # First temporal derivative of pole angle
        self.poleAcc = 0                # Second temporal derivative of pole angle
        self.cartPos = 0                # Position of the cart
        self.cartSpeed = 0              # Velocity of the cart
        self.cartAcc = 0                # Cart acceleration
        self.BB = 10                    # Bang bang force
        self.maxMag = 0.21              # Maximum angle before it fails
        self.minPos = -2.4
        self.maxPos = 2.4
        self.timeStep = 0.02
        self.episodeLength = 300

        self.poleAngle = random.uniform(-self.maxMag, self.maxMag)  # Pole angle (theta)
        self.poleSpeed = 0

        self.actions = [-15, -10, -5, 0, -5, -10, -15]   # Predefined possible actions


    def step(self, action):
        self.BB = action  # If the correct force value is passed (-F to F)
        # Her kommer algoritmen fra prosjekt dokumentet

        # BB force is the variable acting here, pass it as a parameter?

        self.poleAcc = (self.g * np.sin(self.poleAngle) +
                            (np.cos(self.poleAngle)*(-self.BB-self.poleMass*self.length*self.poleAcc*np.sin(self.poleAngle)))
                            /(self.poleMass + self.cartMass))\
                           /(self.length*(4/3 - (self.poleMass*np.cos(self.poleAngle)**2)/(self.poleMass + self.cartMass)))

        self.cartAcc = (self.BB + self.poleMass*self.length*(self.poleSpeed**2*np.sin(self.poleAngle) - self.poleAcc(self.poleAngle) * np.cos(self.poleAngle)))\
                       /(self.poleMass + self.cartMass)

        self.poleSpeed = self.poleSpeed + self.timeStep*self.poleAcc

        self.cartSpeed = self.cartSpeed + self.timeStep*self.cartAcc

        self.poleAngle = self.poleAngle + self.timeStep*self.poleSpeed

        self.cartPos = self.cartPos + self.timeStep*self.cartSpeed

        return self.cartPos, self.cartSpeed, self.poleAngle, self.poleSpeed

    # Returning the possible actions
    def get_actions(self):
        return self.actions

    def get_states(self, ):
        # x: +-0.8, +- 2.4
        # theta: 0, +-1, +-2, +-12
        # cartSpeed: 0.5, inf
        # poleSpeed: 50, inf

        # 4*7*2*2 = 112 states


    def map_discrete_state(self, cs):
        """
        Maps from a continuous state to a discrete state-space
        :param cs:
        :return:
        """



    def optimal_action(self):



