from envs.rom_wrapper import ROMWrapper
from nes_py.wrappers import JoypadSpace
from nes_py._image_viewer import ImageViewer
from nes_py.app.play_human import play_human
from nes_py.app.cli import _get_args
import time


def play_human_custom(env):
    """
    re-using default nes-py play human function
    https://github.com/Kautenja/nes-py/blob/1dda1ad37a84e3ca67dfbebd7cc0c2d8e4cf2489/nes_py/app/play_human.py

    """
    play_human(env)


def play_random_custom(env, steps):
    _NOP = 0

    actions = [
        ['start'],
        ['NOOP'],
        ['right', 'A'],
        ['left', 'A'],
        ['left', 'B'],
        ['right', 'B'],
        ['up'],
        ['down'],
        ['A'],
        ['B']
    ]

    env = JoypadSpace(env, actions)

    env.reset()

    action = 0
    start = time.time()
    # play_human
    for t in range(0, steps):
        # get the mapping of keyboard keys to actions in the environment
        if hasattr(env, 'get_keys_to_action'):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, 'get_keys_to_action'):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            raise ValueError('env has no get_keys_to_action method')

        # # change action every 6 frames
        if t % 6 == 0:
            action = env.action_space.sample()

            # after 500 timesteps, stop pressing start button
            if t > 500:
                while action == 0:
                    action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
        # print("---------------------------t: ", t)
        # print("action space: ", action, env.action_space)
        # print("obs: ", observation)
        # print("reward: ", reward)
        # print("info: ", info)
        # runs game at about 60fps
        time.sleep(0.016667)
        env.render()

    end = time.time()
    env.close()
    print("time: ", (end - start), " seconds  for ", steps, "steps")


class RandomAgent():
    def __init__(self, args):
        # required arg
        rom_path = args.rom

        # create the environment
        env = ROMWrapper(rom_path)

        if args.mode == 'human':
            print("Playing as human")
            play_human_custom(env)
        else:
            print("Playing with random custom agent")
            play_random_custom(env, args.steps)
