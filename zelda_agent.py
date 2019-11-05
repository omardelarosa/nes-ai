from rom_wrapper import ROMWrapper
from nes_py.wrappers import JoypadSpace
import time

ROM_PATH = "./roms/zelda.nes"

"""The main entry point for the command line interface."""
# get arguments from the command line
# create the environment
env = ROMWrapper(ROM_PATH)

actions = [
    ['start'],
    ['NOOP'],
    ['right'],
    ['left'],
    ['left'],
    ['right'],
    ['A'],
    ['B']
]

env = JoypadSpace(env, actions)

env.reset()


action = 0
# play_human
for t in range(0, 5000):
    # favor start during menu screen
    # if t < 500:
    #     action = 0
    # change action every 6 frames
    if t % 6 == 0:
        # while action == 0:
        action = env.action_space.sample()
        if t > 500:
            while action == 0:
                action = env.action_space.sample()

    observation, reward, done, info = env.step(action)
    print("---------------------------t: ", t)
    print("action space: ", action, env.action_space)
    # print("obs: ", observation)
    print("reward: ", reward)
    print("info: ", info)
    # runs game at about 60fps
    time.sleep(0.016667)
    env.render()

env.close()
