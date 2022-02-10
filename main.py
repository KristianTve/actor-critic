from Hanoi import hanoi
import numpy as np
from RL import RL
from gambler import gambler
from CartPole import CartPole

if __name__ == '__main__':
    mode = "cartpole"


    if mode == "cartpole":
        crt = CartPole()
        rl = RL()
        rl.actor_critic(crt.get_state,
                        crt.get_moves,
                        crt.step,
                        crt.reset_problem,
                        crt.is_final,
                        episodes=2200,
                        time_steps=500,
                        lr=0.01)
    if mode == "hanoi":
        hano = hanoi()
        rl = RL()
        rl.actor_critic(hano.get_state,
                        hano.get_moves,
                        hano.step,
                        hano.reset_problem,
                        hano.is_final,
                        200,
                        300,
                        0.001)
    if mode == "gambler":
        gmb = gambler()
        rl = RL()
        rl.actor_critic(gmb.get_state,
                        gmb.get_actions,
                        gmb.step,
                        gmb.reset,
                        gmb.is_final,
                        200,
                        300,
                        0.2)




