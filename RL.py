import numbers
from statistics import mean

import numpy as np
import random

class RL:

    def __init__(self):
        self.P = {}       # Dictionary for values associated with possible STATE & ACTION pairs  (policy eval for actor)
        self.V = {}       # Dictionary for critic evaluations of STATES

        self.aE = {}      # Eligibilities for the actor state, value pairs
        self.cE = {}      # Eligibilities for the critic states

        self.discount = 0.95     # Discount factor  (1 for deterministic environments (Hanoi)
        self.trace_decay = 0.4  # Factor for decaying trace updates
        self.epsilon_original = 0.5      # Epsilon greedy factor probability for choosing a random action
        self.epsilon = 1      # Epsilon greedy factor probability for choosing a random action

    def actor_critic(self, get_state, get_actions, do_action, reset, finished, episodes, time_steps, lr):
        """
        This method should receive the current state and the possible actions as input

        Parameters
        ----------
        @param env   -  The environment model
        @param state  -  The current state
        @param state_size  -  The size of the state space
        @param action_space  - The available actions for the actor
        @param lr - The rate of adjustment when learning
        @param episodes - Number of episodes
        @param time_steps - Number of timesteps in an episode
        """

        init_state = get_state()
        init_actions = get_actions()

        action = 0
        state = init_state
        action_prime = 0
        state_prime = 0
        TD_error = 0
        finished_counter = 0
        runs = 0
        total_runs = episodes * time_steps

        iters_before_finished = []

        min_iter = np.inf
        max_iter = 0

        """*** Initializing V(s) and P(s,a) ***"""
        # Initialize Π(s,a) <-- 0 ∀s,a (actor)
        for action in init_actions:
            self.P[self.keyify(init_state, action)] = 0  # State, action value initialization

        # Initialize V(s) with small random values   (critic)
        self.V[self.keyify(init_state)] = random.uniform(0, 3)
        """************************************"""

        for epi in range(episodes):
            #self.epsilon = self.epsilon_original*((total_runs-runs)/total_runs)      # Decreasing epsilon
            #print(self.epsilon)
            self.epsilon *= 0.97
            print(self.epsilon)

            state_action_buffer = []
            state_buffer = []

            # Reset eligibilities for the actor and critic
            for i in self.P.keys():
                self.aE[i] = 0

            # Reset eligibility with small    (critic)
            for i in self.V.keys():
                self.cE[i] = 0     #

            reset()     # Resets the problem space for a new episode
            state = init_state  # Transfers initial state to recursive variable
            # Selecting best initial action by the policy
            action = self.select_best_action(init_state, init_actions)

            # Repeat for every step of the episode
            for iter in range(time_steps):
                runs += 1
                # Perform action and receive s' and new possible actions
                if iter > max_iter:     # Save longest episode
                    max_iter = iter

                state_prime, new_actions, reward = do_action(action)
                state_buffer.append(self.keyify(state))                     # Saves the state to buffer/trail (path) Keys that have been visited
                state_action_buffer.append(self.keyify(state, action))      # Saves the state and action to buffer/trail (path) Keys that have been visited

                # Initialize policy when new (non existing) (s, a) pairs
                for act in range(len(new_actions)):
                    if not self.keyify(state_prime, new_actions[act]) in self.P:
                        self.P[self.keyify(state_prime, new_actions[act])] = 0  # State, action value initialization
                        #self.aE[self.keyify(state_prime, new_actions[act])] = 0     # Initializing eligibility (Actor)

                if not self.keyify(state, action) in self.P:
                    self.P[self.keyify(state, action)] = 0  # State, action value initialization

                # Initialize critic table for state when it doesnt exist
                if not self.keyify(state_prime) in self.V:
                    self.V[self.keyify(state_prime)] = random.uniform(0, 3)
                    #self.cE[self.keyify(state_prime)] = 0  # Initializing eligibility (critic)

                # ACTOR: a' <-- Π(s') the action dictated by the current policy for state s'
                action_prime = self.select_best_action(state_prime, new_actions)

                # ACTOR: e(s,a) <-- 1 (update eligibility for policy to 1)
                self.aE[self.keyify(state, action)] = 1

                # CRITIC: Calculated temporal difference error (reward + discount*V(s') - V(s))
                TD_error = reward + self.discount*self.V[self.keyify(state_prime)] - self.V[self.keyify(state)]

                # CRITIC e(s) <-- 1 (Eligibility is set to 0 for current state)
                self.cE[self.keyify(state)] = 1

                # For all states and actions in the current episode
                for st in state_buffer:
                    # CRITIC: Calculate new value for state S  V(s) <-- V(s) + lr*TDerror*eligibility(s)
                    self.V[st] += lr * TD_error * self.cE[st]

                    # CRITIC: Calculate new (lower) eligibility for state
                    self.cE[st] = self.discount * self.trace_decay * self.cE[st]  # Decrease eligibility

                for sta in state_action_buffer:
                    # ACTOR: Calculate the new value for Π(s,a)
                    self.P[sta] += lr*TD_error*self.aE[sta]

                    # ACTOR: Calculate the new lower eligibility
                    self.aE[sta] = self.discount*self.trace_decay*self.aE[sta]  #  Decrease eligibility

                # Update s <-- s' and a <-- a'
                state = state_prime
                action = action_prime

                if iters_before_finished:
                    print("Episode: " + str(epi)+" | " + "Iteration: " + str(iter) + " | " + str(finished_counter) + " | Longest Episode: " + str(max_iter) + " | Shortest: " + str(min_iter) + " | Avg finished: " + str(mean(iters_before_finished)))
                else:
                    print("Episode: " + str(epi)+" | " + "Iteration: " + str(iter) + " | " + str(finished_counter) + " | Longest Episode: " + str(max_iter) + " | Shortest: " + str(min_iter))

                #print(state)

                if finished(state):      # Found the solution (s is the end state)
                    if iter < min_iter:
                        min_iter = iter
                    iters_before_finished.append(iter)
                    finished_counter += 1
                    break

    def keyify(self, state, action=None):
        return str(state) if not action else str(state) + str(action)

    def select_best_action(self, state, actions):
        best_action = 0
        best_action_value = -np.inf
        random_num = np.random.uniform(0, 1)

        if not random_num < self.epsilon:
            for action in actions:
                if self.P[self.keyify(state, action)] >= best_action_value:
                    best_action_value = self.P[self.keyify(state, action)]
                    best_action = action

        else:
            return actions[np.random.randint(0, len(actions) - 1)]
        #if not best_action:
        #    print("YE")
        #    return actions[np.random.randint(0, len(actions)-1)]  # Hack (plz remove)
        #print(best_action)

        return best_action

