import numpy as np
import random

class RL:

    def __init__(self):
        self.P = {}      # Dictionary for values associated with possible STATE & ACTION pairs  (policy eval for actor)
        self.V = {}     # Dictionary for critic evaluations of STATES

        self.aE = {}      # Eligibilities for the actor state, value pairs
        self.cE = {}     # Eligibilities for the critic states

        self.discount = 0.99     # Discount factor
        self.trace_decay = 0.85  # Factor for decaying trace updates
        self.epsilon = 0.10      # Epsilon greedy factor probability for choosing a random action

    def actor_critic(self, env, state, state_size, action_space, episodes, time_steps, lr):
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
        # TODO LIST
        # Rework the indexing?

        action = 0   # Current action


        # Initialize Π(s,a) <-- 0 ∀s,a (actor)
        for i in range(state_size):
            for j in range(action_space):
                self.P[i.toString()+","+j.toString()] = random.uniform(0, 3)  # State, action value initialization

        # Initialize V(s) with small random values   (critic)
        for i in range(state_size):
            self.V[i.toString()] = random.uniform(0, 3)  # Action value/probability initialization

        for _ in range(episodes):

            # Reset eligibilities for the actor and critic
            for i in range(state_size):
                for j in range(action_space):
                    self.aE[i.toString() + "," + j.toString()] = 0

            # Reset eligibility with small    (critic)
            for i in range(state_size):
                self.cE[i.toString()] = 0

            # Initialize both state and available actions
            for i in range(state_size):
                for j in range()
            # TODO: Trigger the models initialization for state and actions

            # Repeat for every step of the episode
            for _ in range(time_steps):
                # TODO Do action a from state s, moving the system to state s' and receiving reinforcement r

                # TODO ACTOR: a' <-- Π(s') the action dictated by the current policy for state s'

                # ACTOR: e(s,a) <-- 1 (update eligibility for actor to 1)
                self.aE["1,1"] = 1  # TODO Exchange "1,1" with state, action index

                # TODO CRITIC: Calculated temporal difference error (reward + discount*V(s') - V(s))

                # CRITIC e(s) <-- 1 (Eligibility is set to 0 for current state)
                self.aE["1"] = 1  # TODO Exchange "1" with state index

                # For all states and actions in the current episode
                for i in range(state_size):
                    for j in range(action_space):

                        # CRITIC: Calculate new value for state S  V(s) <-- V(s) + lr*TDerror*eligibility(s)
                        # TODO Calculate TD error
                        self.V[i.toString()] += lr*["TEMPORAL DIFFERENCE ERROR HERE"]*self.cE[i.toString()]

                        # CRITIC: Calculate new (lower) eligibility for state
                        self.cE[i.toString()] = self.discount*self.trace_decay*self.cE[i.toString()]

                        # ACTOR: Calculate the new value for Π(s,a) TODO Calculate TD error
                        self.P[i.toString()+","+j.toString()] += lr*["TD ERROR HERE"]*self.aE[i.toString()+","+j.toString()]

                        # ACTOR: Calculate the new lower eligibility
                        self.aE[i.toString()+","+j.toString()] = self.discount*self.trace_decay*self.aE[i.toString()+","+j.toString()]

                # TODO Update s <-- s' and a <-- a'
                # state = newState
                # action = newAction

                # TODO is S end state?
                # Check with model





