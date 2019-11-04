from megaman_env import MegamanEnv
import time

ROM_PATH = "./roms/megaman2.nes"

"""The main entry point for the command line interface."""
# get arguments from the command line
# create the environment
env = MegamanEnv(ROM_PATH)

env.reset()

# play_human
for t in range(0, 1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print("---------------------------t: ", t)
    print("action space: ", action)
    # print("obs: ", observation)
    print("reward: ", reward)
    print("info: ", info)
    # runs game at about 60fps
    time.sleep(0.016667)
    env.render()

env.close()
