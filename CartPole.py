import math
import random
import numpy as np
from config_manager import config_manager
class CartPole:

    def __init__(self):
        config = config_manager()


        self.length = 0.5
        self.poleMass = 0.1
        self.cartMass = 1
        self.g = -9.8                    # Gravity

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

        self.poleAngle_bin = 0
        self.poleSpeed_bin = 0
        self.cartPos_bin = 0
        self.cartSpeed_bin = 0


        self.poleAngle = random.uniform(-self.maxMag+0.1, self.maxMag-0.1)  # Pole angle (theta)
        self.poleSpeed = 0

        self.actions = [-15, -10, -5, 0, 5, 10, 15]   # Predefined possible actions
        #self.actions = [-10, 0, 10]
        #self.actions = [-self.BB, self.BB]
        #self.actions.append(-self.BB)
        #self.actions.append(self.BB)

        (self.length,
         self.poleMass,
         self.g,
         self.timeStep) = config.get_cartpole_params()


    def step(self, action):
        self.BB = action  # If the correct force value is passed (-F to F)
        # Her kommer algoritmen fra prosjekt dokumentet

        # BB force is the variable acting here, pass it as a parameter?

        self.poleAcc = (self.g * np.sin(self.poleAngle) +
                            (np.cos(self.poleAngle)*(-self.BB-self.poleMass*self.length*self.poleSpeed**2*np.sin(self.poleAngle)))
                            /(self.poleMass + self.cartMass))\
                           /(self.length*(4/3 - (self.poleMass*np.cos(self.poleAngle)**2)/(self.poleMass + self.cartMass)))

        self.cartAcc = (self.BB + self.poleMass*self.length*(self.poleSpeed**2*np.sin(self.poleAngle) - self.poleAcc * np.cos(self.poleAngle)))\
                       /(self.poleMass + self.cartMass)

        self.poleSpeed = self.poleSpeed + self.timeStep*self.poleAcc

        self.cartSpeed = self.cartSpeed + self.timeStep*self.cartAcc

        self.poleAngle = self.poleAngle + self.timeStep*self.poleSpeed

        self.cartPos = self.cartPos + self.timeStep*self.cartSpeed

        reward = self.reward()

        return self.get_state(), self.get_moves(), reward


    # Returning the possible actions
    def get_moves(self):
        return self.actions

    def get_continuous_state(self):
        return [self.poleAngle, self.poleSpeed, self.cartPos, self.cartSpeed]

    def get_state(self):
        state = None  # The state object to be saved into
        # x: +-0.8, +- 2.4
        # theta: 0, +-1, +-2, +-12
        # cartSpeed: 0.5, inf
        # poleSpeed: 50, inf

        self.poleAngle_bin = np.digitize(self.poleAngle, [-0.24, -0.21, -0.18, -0.14, -0.10, -0.06, -0.03, 0, 0.03, 0.06, 0.10, 0.14, 0.18, 0.21, 0.24]) # 15 - 7 (best)
        self.poleSpeed_bin = np.digitize(self.poleSpeed, [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])  # 11 - 6 (best)
        self.cartPos_bin = np.digitize(self.cartPos,     [-2.4, -1.5, -0.8, 0, 0.8, 1.5, 2.4])          # 7  4 (best)
        self.cartSpeed_bin = np.digitize(self.cartSpeed, [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])  # 11 - 6 (best)

        state = [self.poleAngle_bin, self.poleSpeed_bin, self.cartPos_bin, self.cartSpeed_bin]
        return np.array(state)

    def reward(self):
        # reward = 0
        # if self.poleAngle_bin == 0:
        #     reward = 1
        # elif self.poleAngle_bin in [1, 2, 3, 4, 5]:
        #     reward = 0
        # else:
        #     reward = -10
        if self.cartPos_bin > 4:
            cartPosDiff = 7 - self.cartPos_bin  # Offset
        elif self.cartPos_bin <= 4:
            cartPosDiff = 4 - self.cartPos_bin  # Offset


        if self.poleAngle_bin > 7:
            poleAngleDiff = 15 - self.poleAngle_bin   # Offset
        elif self.poleAngle_bin <= 7:
            poleAngleDiff = 7 - self.poleAngle_bin    # Offset

        # reward = 12 - poleAngleDiff - cartPosDiff
        # return poleAngleDiff
        # return (1 - (self.cartPos ** 2) / 11.52 - (self.poleAngle ** 2) / 288)
        #
        ### THE GOLDEN BOI ###
        if abs(self.poleAngle) < 0.03:
           return 0 if self.poleSpeed < 0 else 1
        else:
           return 0 if self.poleAngle < 0 else 1

        # if abs(poleAngleDiff) < 1:
        #     return 0 if poleSpeedDiff > 0 else 1
        # else:
        #     return 0 if poleAngleDiff > 0 else 1
        #return reward


    def reset_problem(self):
        self.poleAngle = random.uniform(-self.maxMag, self.maxMag)  # Pole angle (theta)
        self.poleSpeed = 0              # First temporal derivative of pole angle
        self.poleAcc = 0                # Second temporal derivative of pole angle

        self.cartPos = 0                # Position of the cart
        self.cartSpeed = 0              # Velocity of the cart
        self.cartAcc = 0                # Cart acceleration

    def is_final(self, state=None):   # Continuous problem
        if np.abs(self.poleAngle) > self.maxMag:
            return True
        else:
            if np.abs(self.cartPos) >= self.maxPos:
                return True
            else:
                return False

    def get_continuous_poleAngle(self):
        return self.poleAngle





