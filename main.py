from Hanoi import hanoi
import numpy as np
from RL import RL
from gambler import gambler

if __name__ == '__main__':
    mode = "hanoi"


    if mode == "cartpole":
        pass
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
                        0.1)
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




