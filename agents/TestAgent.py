import gym
import time

ROM_PATH = "../roms/megaman2.nes"


class TestAgent():
    def __init__(self, args):
        env = gym.make("SpaceInvaders-v0")
        observation = env.reset()
        for t in range(1000):
            env.render()
            action = env.action_space.sample()  # your agent here (this takes random actions)
            observation, reward, done, info = env.step(action)
            print("---------------------------t: ", t)
            print("action space: ", env.action_space)
            # print("obs: ", observation)
            print("reward: ", reward)
            print("info: ", info)
            # runs game at about 60fps
            time.sleep(0.016667)
            if done:
                observation = env.reset()
        env.close()
