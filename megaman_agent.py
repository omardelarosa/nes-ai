from rom_wrapper import ROMWrapper
from nes_py.wrappers import JoypadSpace
import time

ROM_PATH = "./roms/megaman2.nes"

"""The main entry point for the command line interface."""
# get arguments from the command line
# create the environment
env = ROMWrapper(ROM_PATH)

actions = [
    ['start'],
    ['NOOP'],
    ['right', 'A'],
    ['left', 'A'],
    ['left', 'B'],
    ['right', 'B'],
    ['A'],
    ['B']
]

env = JoypadSpace(env, actions)

env.reset()


action = 0
# play_human
for t in range(0, 5000):
    # change action every 6 frames
    if t % 6 == 0:
        action = env.action_space.sample()

        # after 500 timesteps, stop pressing start button
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
