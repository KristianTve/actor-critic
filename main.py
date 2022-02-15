from Hanoi import hanoi
import numpy as np
from NNCritic import RLNN
from RL import RL
from gambler import gambler
from CartPole import CartPole
from config_manager import config_manager
if __name__ == '__main__':
    config = config_manager()
    mode, NN = config.get_main_params()

    if NN:
        if mode == "cartpole":
            crt = CartPole()
            rl = RLNN("cartpole")
            rl.actor_critic(crt.get_state,
                            crt.get_moves,
                            crt.step,
                            crt.reset_problem,
                            crt.is_final,
                            episodes=2800,
                            time_steps=500,
                            lr=0.001,
                            get_continous_state=crt.get_continuous_poleAngle)
        if mode == "hanoi":
            hano = hanoi()
            rl = RLNN("hanoi")
            rl.actor_critic(hano.get_state,
                            hano.get_moves,
                            hano.step,
                            hano.reset_problem,
                            hano.is_final,
                            2200,
                            300,
                            0.5)
    else:
        if mode == "cartpole":
            crt = CartPole()
            rl = RL("cartpole")
            rl.actor_critic(crt.get_state,
                            crt.get_moves,
                            crt.step,
                            crt.reset_problem,
                            crt.is_final,
                            episodes=2800,
                            time_steps=500,
                            lr=0.001,
                            get_continous_state=crt.get_continuous_poleAngle)
        if mode == "hanoi":
            hano = hanoi()
            rl = RL("hanoi")
            rl.actor_critic(hano.get_state,
                            hano.get_moves,
                            hano.step,
                            hano.reset_problem,
                            hano.is_final,
                            2200,
                            300,
                            0.5)
        if mode == "gambler":
            gmb = gambler()
            rl = RL("gambler")
            rl.actor_critic(gmb.get_state,
                            gmb.get_actions,
                            gmb.step,
                            gmb.reset,
                            gmb.is_final,
                            episodes=25000,
                            time_steps=300,
                            lr=0.2)




