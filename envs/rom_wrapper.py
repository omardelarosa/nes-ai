"""An OpenAI Gym interface to the NES game MEGAMAN2"""
from nes_py import NESEnv
import numpy as np


"""
This is a generic wrapper based on the example on nes_py wiki
"""


class ROMWrapper(NESEnv):
    """An OpenAI Gym interface to the NES game MEGAMAN2"""

    def __init__(self, rom_path):
        """Initialize a new MEGAMAN2 environment."""
        super(ROMWrapper, self).__init__(rom_path)
        self.last_screen = None
        self.screen_distance = 0
        self.step_num = 0
        self.score = 0.0

        # setup any variables to use in the below callbacks here

    def _will_reset(self):
        """Handle any RAM hacking after a reset occurs."""
        # use this method to perform setup before and episode resets.
        # the method returns None
        pass

    def _did_reset(self):
        """Handle any RAM hacking after a reset occurs."""
        # use this method to access the RAM of the emulator
        # and perform setup for each episode.
        # the method returns None
        pass

    def _did_step(self, done):
        """
        Handle any RAM hacking after a step occurs.

        Args:
            done: whether the done flag is set to true

        Returns:
            None

        """

        # TWO Ways of analyzing state:

        # 1. as RAM: array
        # print("RAM: ", self.ram)  # debugging RAM
        # print("MEGAMAN SCORE: ", self.ram[114:120])
        score_vec = self.ram[114:120]
        self.score = 0.0
        for i in range(0, len(score_vec)):
            self.score = self.score + (score_vec[i] * 10**i)
        # print("IntScore: ", self.score)
        # 2. as Screen Image Array

        # https://github.com/Kautenja/nes-py/blob/master/nes_py/nes_env.py#L69
        # shape of the screen as 32-bit RGB (C++ memory arrangement)
        ## SCREEN_SHAPE_32_BIT = SCREEN_HEIGHT, SCREEN_WIDTH, 4
        # print("Screen: ", self.screen.shape)  # debugging screen
        if self.step_num > 0:
            a = np.array(self.screen.flatten())
            b = self.last_screen
            self.screen_distance = np.linalg.norm(a - b)
        self.last_screen = np.array(self.screen.flatten())
        # print("Screen Distance: ", self.screen_distance)
        self.step_num = self.step_num + 1
        pass

    def _get_reward(self):
        """Return the reward after a step occurs."""
        screen_dist = (self.screen_distance * 0.0000001)
        score = (self.score * 0.001)
        return max(screen_dist, score)

    def _get_done(self):
        """Return True if the episode is over, False otherwise."""
        return False

    def _get_info(self):
        """Return the info after a step occurs."""
        return {
            'ram': np.array(self.ram)
        }


# explicitly define the outward facing API for the module
__all__ = [ROMWrapper.__name__]
