from Hanoi import hanoi
import numpy as np
#from RL import RL

if __name__ == '__main__':
    hano = hanoi()
    #rl = RL()
    #rl.actor_critic(hano.get_state, hano.get_moves, 64, 120, 50, 200, 0.2, hano.reset_problem, step, hano.is_final)
    action = [2, 3, 1]
    peg = np.zeros((4, 3))
    peg[0, 0] = 1
    peg[1, 0] = 2
    peg[2, 0] = 3
    peg[3, 0] = 4

    A = str(peg.flatten())

    #for i in range(10):
    #    hano.step(hano.get_moves()[0])
    #    hano.print_problem()

