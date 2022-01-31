import math
import numpy as np


class hanoi:

    def __init__(self):
        # hanoi big boi
        self.n_pegs = 3  # Number of positions
        self.n_discs = 4  # Number of discs
        self.n_states = self.n_pegs * self.n_discs  # Total combinations possible (states)

        """
        Looper gjennom finner 0 g
        [1  0  0]
        [2  0  0]
        [3  0  0]
        [4  0  0]        
        """
        self.peg = np.zeros((self.n_discs, self.n_pegs))
        self.peg[0, 0] = 1
        self.peg[1, 0] = 2
        self.peg[2, 0] = 3
        self.peg[3, 0] = 4

        # Final state:
        self.final = np.zeros((self.n_discs, self.n_pegs))
        self.final[0, self.n_pegs-1] = 1
        self.final[1, self.n_pegs-1] = 2
        self.final[2, self.n_pegs-1] = 3
        self.final[3, self.n_pegs-1] = 4

    def step(self, action):
        disc =    action[0]     # Extracting disc number
        toPeg =   action[2]     # Extracting end peg

        reward = self.reward(self.peg)
        # Each step is punished by 1 (or float)
        # If steps > X - game is over (300 steps)
        # Give large reward when completing
        # Proportional reward with disc size on last pole 4 points for disc 4, 3 for disc 3 etc..

        # Perform action requested: [disc, fromPeg, toPeg]
        self.remove_disc(disc)       # Removes disc from original position
        self.put_disc(toPeg, disc)  # Puts disc on given peg (first available spot)

        #Return available actions and reward!!
        return self.peg, self.get_moves(), reward

    def put_disc(self, endPeg, disc):
        """
        Putting the disc in the lowest slot
        """
        for d in reversed(range(self.n_discs)):     # Iterates bottom up for first available slot
            if self.peg[d, endPeg] == 0:    # First available slot
                self.peg[d, endPeg] = disc  # Put disc
                break                       #

    def remove_disc(self, disc):
        self.peg[self.peg == disc] = 0

    def reward(self, state):
        return 0  # Insert intelligent reward logic here

    # Check if the state is final
    def is_final(self, state):
        if state == self.final:
            return True
        else:
            return False


    def get_moves(self):
        """
        Move:
        [Disc, FromPeg, ToPeg]
        #AuxPeg er den siste peggen som ikke blir brukt
        [4,2,3]
        """

        moves = []          # Array for storing the available moves
        discsToMove = []    # Helper array for saving discs that is movable

        for i in range(self.n_pegs):  # Loop through all n_pegs
            for j in range(self.n_discs):  # Loop through all possible slots in peg
                if self.peg[j, i] != 0:  # Is there a disc?
                    discToMove = int(self.peg[j, i])  # Saving disc
                    discsToMove.append((discToMove, i))  # Saving the disc number and peg
                    break  # Only take the top disc

        for disc in discsToMove:
            for k in range(self.n_pegs):  # Loop through all other poles
                move = []               # Reset buffer
                if disc[1] != k:        # Not start pole

                # Find possible end pole
                    for j in range(self.n_discs):
                        if self.peg[j, k] != 0:  # Not empty
                            if self.peg[j, k] > disc[0]:  # The peg contains a disc that is larger
                                endPeg = k
                                move = [disc[0], disc[1], endPeg]
                            break
                        if j == 3:  # If not found any (reached bottom of array)
                            endPeg = k
                            move = [disc[0], disc[1], endPeg]
                            break
                    if move:
                        moves.append(move)

        return moves

    def get_state(self):
        return self.peg

    def print_problem(self):
        print(self.peg)

    def reset_problem(self):
        self.peg = np.zeros((self.n_discs, self.n_pegs))
        self.peg[0, 0] = 1
        self.peg[1, 0] = 2
        self.peg[2, 0] = 3
        self.peg[3, 0] = 4

