import copy
import numbers
from statistics import mean

import keras.optimizer_experimental.sgd
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from config_manager import config_manager
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple


class RLNN:

    def __init__(self, mode):
        self.config = config_manager()

        self.mode = mode
        self.P = {}  # Dictionary for values associated with possible STATE & ACTION pairs  (policy eval for actor)

        self.aE = {}  # Eligibility for the actor state, value pairs

        #self.discount = 1  # Discount factor  (1 for deterministic environments (Hanoi)
        #self.trace_decay = 0.5  # Factor for decaying trace updates (HANOI: 0.5)
        #self.epsilon = 1  # Epsilon greedy factor probability for choosing a random action

        self.runs = 0
        self.epi = 0

        self.arrayE = []  # Episodes
        self.arrayR = []  # Runs before completion
        self.arrayPA = []
        self.episode_PA = []  # Buffer for storing current episodes pole angle (for illustration only)
        self.NN = True  # Change this with config later

        self.continuous_state = None

        (self.layers,
         self.input_neurons,
         self.batch_size,
         self.verbose,
         self.layer_size,
         self.layer_act,
         self.optimizer) = self.config.fetch_net_data()

        (self.critic_lr,
         self.actor_lr,
         self.discount,
         self.trace_decay,
         self.epsilon,
         self.episodes,
         self.time_steps) = self.config.fetch_actor_critic_data()

        self.critic = NN(mode=mode,
                         layers=self.layers,
                         input_size=self.input_neurons,
                         layer_size=self.layer_size,
                         layer_act=self.layer_act,
                         optimizer=self.optimizer,
                         critic_lr=self.critic_lr)

    def actor_critic(self, get_state, get_actions, do_action, reset, finished, episodes, time_steps, lr,
                     get_continous_state=None):
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

        self.continuous_state = get_continous_state
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

        training_cases = []

        min_iter = np.inf
        max_iter = 0

        """*** Initializing V(s) and P(s,a) ***"""
        # Initialize Π(s,a) <-- 0 ∀s,a (actor)
        self.initialize_actor(state, init_actions)
        """************************************"""

        for epi in range(self.episodes):
            self.runs = 0
            self.epi += 1
            if self.epsilon >= 0.001:
                self.epsilon *= 0.97  # Degrading the epsilon value for each episode
            print(self.epsilon)
            state_action_buffer = []
            state_buffer = []
            training_cases = []
            TD_error_buffer = []
            self.episode_PA = []

            # Reset eligibilities for the actor and critic
            for i in self.P.keys():
                self.aE[i] = 0

            if not epi == 0:  # If any episode has been run already
                reset()  # Resets the problem space for a new episode
                state = get_state()  # Transfers initial state to recursive variable
                init_actions = get_actions()
                self.initialize_actor(state, init_actions)
            else:
                state = init_state  # If it is first run, use initial state

            # Selecting best initial action by the policy
            action = self.select_best_action(state, init_actions)

            # Repeat for every step of the episode
            for iter in range(self.time_steps):
                self.runs += 1  # Just informative variable

                # Perform action and receive s' and new possible actions
                if iter > max_iter:  # Save longest episode
                    max_iter = iter

                state_prime, new_actions, reward = do_action(action)  # Step function in the environments

                state_buffer.append(
                    self.keyify(state))  # Saves the state to buffer/trail (path) Keys that have been visited
                state_action_buffer.append(self.keyify(state,
                                                       action))  # Saves the state and action to buffer/trail (path) Keys that have been visited

                # Initialize policy when new (non existing) (s, a) pairs
                for act in range(len(new_actions)):
                    if not self.keyify(state_prime, new_actions[act]) in self.P:
                        self.P[self.keyify(state_prime, new_actions[act])] = 0  # State, action value initialization
                        self.aE[self.keyify(state_prime, new_actions[act])] = 0  # Initializing eligibility (Actor)

                if not self.keyify(state, action) in self.P:
                    self.P[self.keyify(state, action)] = 0  # State, action value initialization

                # ACTOR: a' <-- Π(s') the action dictated by the current policy for state s'
                if len(new_actions):
                    action_prime = self.select_best_action(state_prime, new_actions)

                # ACTOR: e(s,a) <-- 1 (update eligibility for policy to 1)
                self.aE[self.keyify(state, action)] = 1

                # CRITIC: V(s')(predicted) and V(s)(true)
                if self.mode == "gambler":
                    V_s_true = reward + self.discount * self.critic.predict(int(float(state_prime))).numpy()[0][0]
                    training_cases.append((int(float(state)), V_s_true))  # Adding case (features, target)
                else:
                    V_s_true = reward + self.discount * self.critic.predict(state_prime).numpy()[0][0]
                    training_cases.append((state, V_s_true))  # Adding case (features, target)

                TD_error_buffer.append(V_s_true)

                if self.runs % self.batch_size == 0 and self.runs != 0:
                    # Train critic network on cases gathered during episode
                    loss = self.critic.train_cases(training_cases)
                    loss = loss.history['loss'][0]  # Extract loss as float
                    for TD_error in TD_error_buffer:
                        for sta in state_action_buffer:

                            # ACTOR: Calculate the new value for Π(s,a)
                            self.P[sta] += self.actor_lr * TD_error * self.aE[sta]

                            # ACTOR: Calculate the new lower eligibility
                            self.aE[sta] = self.discount * self.trace_decay * self.aE[sta]  # Decrease eligibility
                    TD_error_buffer = []
                    training_cases = []  # Reset training cases
                # Update s <-- s' and a <-- a'
                state = copy.deepcopy(state_prime)
                action = copy.deepcopy(action_prime)

                if iters_before_finished:
                    print("Episode: " + str(epi) + " | " + "Iteration: " + str(iter) + " | " + str(
                        finished_counter) + " | Longest Episode: " + str(max_iter) + " | Shortest: " + str(
                        min_iter) + " | Avg finished: " + str(mean(iters_before_finished)))
                else:
                    print("Episode: " + str(epi) + " | " + "Iteration: " + str(iter) + " | " + str(
                        finished_counter) + " | Longest Episode: " + str(max_iter) + " | Shortest: " + str(min_iter))

                self.episode_PA.append(self.continuous_state())

                if finished(state):  # Found the solution (s is the end state)
                    if iter < min_iter:
                        min_iter = iter
                    iters_before_finished.append(iter)
                    finished_counter += 1

                    self.arrayE.append(int(self.epi))
                    self.arrayR.append(int(self.runs))
                    if self.mode == "cartpole":
                        if len(self.episode_PA) > len(self.arrayPA):
                            self.arrayPA = self.episode_PA     # Storing longest episode pole angle

                    break

                if iter == 299:
                    self.arrayE.append(int(self.epi))
                    self.arrayR.append(int(self.runs))
                    if self.mode == "cartpole":
                        if len(self.episode_PA) > len(self.arrayPA):
                            self.arrayPA = self.episode_PA      # Storing longest episode pole angle

            if epi % 50 == 0:  # Print func boi
                self.mode_selector()
        if self.mode == "cartpole":
            self.print_cartpole()  # Printing pole angle

    def keyify(self, state, action=None):
        return str(state) if not action else str(state) + str(action)

    def select_best_action(self, state, actions):
        best_action = 0  # Buffer for storing best action
        best_action_value = -np.inf  # (Buffer) Mechanism for selecting a better policy than -inf
        random_num = np.random.uniform(0, 1)  # Random number for epsilon greedy mechanism
        np.random.shuffle(actions)
        if not random_num < self.epsilon:
            for action in actions:
                if self.P[self.keyify(state, action)] >= best_action_value:
                    best_action_value = self.P[self.keyify(state, action)]
                    best_action = action

        else:
            return actions[np.random.randint(0, len(actions) - 1)] if (len(actions) != 1) else actions[0]

        return best_action

    def print_cartpole(self):
        plt.plot(self.arrayPA)
        plt.show()


    def print_hanoi(self):
        plt.plot(self.arrayE, self.arrayR)
        plt.show()

    def print_gambler(self):
        array = []
        for state in np.arange(1, 100):
            state = str(state) + "."
            best = 0
            best_action = 0
            for action in np.arange(1, 100):
                if self.keyify(state, action) in self.P:
                    if self.P[self.keyify(state, action)] > best:
                        best = self.P[self.keyify(state, action)]
                        best_action = action
            array.append(best_action)
        plt.plot(array)
        plt.show()

    def mode_selector(self):
        if self.mode == "cartpole":
            self.print_hanoi()
        elif self.mode == "hanoi":
            self.print_hanoi()
        elif self.mode == "gambler":
            self.print_gambler()

    # Initializes the dictionaries for actor and critic
    def initialize_actor(self, state, actions):
        for action in actions:
            if not self.keyify(state, action) in self.P:
                self.P[self.keyify(state, action)] = 0  # State, action value initialization


class NN:

    def __init__(self,
                 mode,
                 layers,
                 input_size,
                 layer_size,
                 layer_act,
                 optimizer,
                 critic_lr):

        self.mode = mode
        self.optimizer = optimizer
        self.critic_lr = critic_lr

        self.mode = mode

        # Input layer generation
        inputs = tf.keras.Input(shape=(input_size,))

        # Hidden layer generation
        x = inputs
        for i in range(layers):
            if layer_act[i] == "relu":
                x = tf.keras.layers.Dense(layer_size[i], activation=tf.nn.relu)(x)
            elif layer_act[i] == "tanh":
                x = tf.keras.layers.Dense(layer_size[i], activation=tf.nn.tanh)(x)
            elif layer_act[i] == "sigmoid":
                x = tf.keras.layers.Dense(layer_size[i], activation=tf.nn.sigmoid)(x)
            else:
                x = tf.keras.layers.Dense(layer_size[i], activation=tf.nn.relu)(x)

        # Output layer generation
        if layer_act[layers] == "sigmoid":
            outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)
        elif layer_act[layers] == "relu":
            outputs = tf.keras.layers.Dense(1, activation=tf.nn.relu)(x)
        elif layer_act[layers] == "tanh":
            outputs = tf.keras.layers.Dense(1, activation=tf.nn.tanh)(x)
        else:
            outputs = tf.keras.layers.Dense(1, activation=None)(x)

        # Creating the model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Setting optimizer and compiling
        if self.optimizer == "sgd":
            self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.critic_lr, name="SGD"),
                               loss=tf.keras.metrics.mean_squared_error)
        if self.optimizer == "adam":
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.critic_lr),
                               loss=tf.keras.metrics.mean_squared_error)

    def predict(self, state):
        state = np.array(state)
        # for state in states:
        if state.ndim == 2:
            return self.model(tf.convert_to_tensor([state.flatten()]))
        else:
            return self.model(tf.convert_to_tensor([state]))

    def train_cases(self, cases):
        cases_flat = []
        targets = []
        for case in cases:
            if isinstance(case[0], int):
                cases_flat.append(case[0])
            elif case[0].ndim == 2:
                cases_flat.append(case[0].flatten())
            else:
                cases_flat.append(case[0])
            targets.append(case[1])

        loss = self.model.fit(tf.convert_to_tensor(cases_flat), tf.convert_to_tensor(targets))
        return loss

    # loss function and its derivative
    def mse(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))
