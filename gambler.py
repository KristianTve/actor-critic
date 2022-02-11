import math
import random

import numpy as np

class gambler:

    def __init__(self, win_prob=0.4):
        self.wager = 0            # Amount of waged money
        self.win_prob = win_prob  # Chance of winning
        self.victory = 100        # Victory limit
        self.failure = 0

        self.money = random.randint(1, 99)

    def step(self, action):
        self.wager = action

        print("Start Wager: " + str(self.wager))
        print("Start Money: " + str(self.money))

        if self.coin_flip():
            self.money += self.wager        # Gets dat money back man
            print("WIN")
        else:
            self.money -= self.wager        # Looses the money
            print("loss")

        reward = self.get_reward()

        print("Money: " + str(self.money))

        return self.get_state(), self.get_actions(), reward

    def get_actions(self):
        # Cannot pick more than it takes to win
        max = 100 - self.money
        if max > self.money:
            max = self.money

        bets = np.arange(1, max+1)
        np.random.shuffle(bets)

        return bets

    def get_state(self):
        #return self.money
        return str(self.money)+"."

    def coin_flip(self):
        return True if np.random.uniform(0, 1) < self.win_prob else False

    # Reward function
    def get_reward(self):
        return 1 if self.money == 100 else 0

    def is_final(self, state):
        return True if self.money == 100 or self.money == 0 else False

    def reset(self):
        self.money = random.randint(1, 99)
        self.wager = 0
