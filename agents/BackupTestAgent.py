from envs.rom_wrapper import ROMWrapper
from nes_py.wrappers import JoypadSpace
from nes_py._image_viewer import ImageViewer
from nes_py.app.play_human import play_human
from nes_py.app.cli import _get_args
import time
import gym
import math

class BackupTestAgent():
    def __init__(self, args):
        rom_path = args.rom

        actions = [
                    ['start'],
                    ['NOOP'],
                    ['right', 'A'],
                    ['left', 'A'],
                    ['left', 'B'],
                    ['right', 'B'],
                    ['down', 'A'],
                    ['down', 'B'],
                    ['up', 'A'],
                    ['up', 'B'],
                    ['up'],
                    ['down'],
                    ['A'],
                    ['B']
                ]

        self.env = ROMWrapper(rom_path)
        self.env = JoypadSpace(self.env, actions)
        observation = self.env.reset()
        for i in range(5):
            if i !=0:
                observation = self.env.reset()
                print("Reset After Backup")
            for t in range(1000):
                self.env.render()
                action = self.env.action_space.sample()  # your agent here (this takes random actions)
                observation, reward, done, info = self.env.step(action)
                print("---------------------------t: ", t)
                print("action space: ", self.env.action_space)
                # print("obs: ", observation)
                print("reward: ", reward)
                print("info: ", info)
                # runs game at about 60fps
                time.sleep(0.016667)
                if t % 750 == 0:
                    self.env.backup()
                    print("state backed up")
                if done:
                    observation = self.env.reset()
        self.env.close()
