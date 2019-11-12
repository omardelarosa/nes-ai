from envs.rom_wrapper import ROMWrapper
from nes_py.wrappers import JoypadSpace
from nes_py._image_viewer import ImageViewer
from nes_py.app.play_human import play_human
from nes_py.app.cli import _get_args
import time
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt


def play_human_custom(env):
    """
    re-using default nes-py play human function
    https://github.com/Kautenja/nes-py/blob/1dda1ad37a84e3ca67dfbebd7cc0c2d8e4cf2489/nes_py/app/play_human.py

    """
    play_human(env)


class HillClimberAgent:
    def __init__(self, args):
        # required arg
        rom_path = args.rom

        # create the environment
        env = ROMWrapper(rom_path)

        self.actions = [
            ['start'],
            ['NOOP'],
            ['right', 'A'],
            ['left', 'A'],
            ['left', 'B'],
            ['right', 'B'],
            ['up'],
            ['down'],
            ['left'],
            ['right'],
            ['A'],
            ['B']
        ]

        self.action_to_num = {
            '-'.join(a for a in possible): index
            for index, possible in enumerate(self.actions)
        }

        if args.mode == 'human':
            print("Playing as human")
            play_human_custom(env)
        else:
            print("Playing with hill climber")
            self.play_hill_climber(env, args.steps)


    def manual_menu_solve(self, env):
        print('manually solving menu')
        env.step(self.action_to_num['NOOP'])
        env.render()
        print('go')

        manual_solve = ['NOOP',
                        'start',
                        'NOOP',
                        'NOOP',
                        'left',
                        'NOOP',
                        'left',
                        'start',
                        'NOOP',
                        'NOOP']
        solve_step = 0
        frames = 500
        change_every = int(frames / len(manual_solve)) + 1
        print('change_every', change_every)
        for t in range(frames):
            if t % change_every == 0:
                step = manual_solve[solve_step]
                act_num = self.action_to_num[step]
                solve_step += 1
                # print('pressing', step, act_num)
            env.step(act_num)
            env.render()
        print('speeding through wait')
        loading_screen_frames = 500
        for t in range(loading_screen_frames):
            env.step(self.action_to_num['NOOP'])
            env.render()
        print('playing game')


    def get_max_shift(self, poi1, poi2):
        x_range = int(poi1.shape[1] / 10)
        y_range = int(poi1.shape[0] / 10)
        outs = []
        out_coords = []
        for x in range(-1 * x_range, x_range):
            for y in range(-1 * y_range, y_range):
                poi2_tmp = np.roll(poi2, x, axis=1)
                poi2_tmp = np.roll(poi2_tmp, y, axis=0)
                count = np.sum(poi1 & poi2_tmp)
                outs.append(count)
                out_coords.append((x, y))

        shift_index = outs.index(max(outs))
        return out_coords[shift_index]

    def combine_images_for_shift(self, img1, img2, x, y, alpha):
        new_x = img1.shape[1] + abs(x)
        new_y = img1.shape[0] + abs(y)
        pad_img_1 = np.zeros((new_y, new_x), dtype=np.float32)
        pad_img_1[:img1.shape[0], :img1.shape[1]] = img1
        pad_img_2 = np.zeros((new_y, new_x), dtype=np.float32)
        pad_img_2[y:y + img2.shape[0], x:x + img2.shape[1]] = img2
        out_img = cv2.addWeighted(pad_img_1, alpha, pad_img_2, 1 - alpha, 0.0)
        return out_img


    def play_hill_climber(self, env, steps):
        _NOP = 0
        steps = 100
        env = JoypadSpace(env, self.actions)

        change_button_interval = 6 # every 6 steps
        actions_in_sequence = int(steps / change_button_interval) + 1

        best_action_sequence = [self.sample_no_start(env) for _ in range(actions_in_sequence)]
        env.reset()
        best_score = self.evaluate_action_sequence(env, steps, change_button_interval, best_action_sequence)
        while True:
            env.reset()
            new_action_sequence = self.get_modified_actions(env, best_action_sequence, 0.2)
            new_score = self.evaluate_action_sequence(env, steps, change_button_interval, new_action_sequence)
            print('eval seq:', new_action_sequence)
            print('got score:', new_score, 'vs best score:', best_score)
            if new_score > best_score:
                best_score, best_action_sequence = new_score, new_action_sequence

        env.close()

    def get_modified_actions(self, env, best_action_sequence, dropout_probability):
        modified_actions = []
        for action in best_action_sequence:
            if random.random() < dropout_probability:
                modified_actions.append(action)
            else:
                modified_actions.append(self.sample_no_start(env))
        return modified_actions

    def evaluate_action_sequence(self, env, steps, change_button_interval, action_sequence):
        self.manual_menu_solve(env)

        action = 0
        start = time.time()
        screens = []
        action_iter = 0
        score = 0

        # play_human
        for t in range(steps):
            if t % change_button_interval == 0:
                action = action_sequence[action_iter]
                action_iter += 1
                # print('changed to ', self.actions[action], action)

            observation, reward, done, info = env.step(action)
            # runs game at about 60fps
            time.sleep(0.016667)
            env.render()

            if t % 50 == 0:
                screen = self.get_simple_screen(env)
                screens.append(screen)
            if (t % 500 == 0 and t > 0) or t == steps - 1:
                def get_poi(img):
                    img = np.copy(img)
                    gray = np.float32(img)
                    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
                    mask = dst > 0.01 * dst.max()
                    return mask

                print('stitching')
                # is grayscale here
                shifted = screens[0]

                poi1 = get_poi(shifted)
                for i in range(1, len(screens)):
                    poi2 = get_poi(screens[i])
                    x, y = self.get_max_shift(poi1, poi2)
                    print('shift: ', x, y)
                    try:
                        shifted = self.combine_images_for_shift(shifted, screens[i], x, y, 0.7)
                    except Exception:
                        print('exception combining')

                    # cv2.imshow('image_{}'.format(i), shifted)

                score = shifted.shape[0] * shifted.shape[1]
                print('score', score)

                # cv2.waitForKey(0)
                # grayscale: what is wrong?
        end = time.time()
        print("time: ", (end - start), " seconds  for ", steps, "steps")
        return score

    def sample_no_start(self, env):
        action = env.action_space.sample()
        while action == 0:  # no start button
            action = env.action_space.sample()
        return action

    def get_simple_screen(self, env, grayscale=True):
        color_screen = env.render(mode='rgb_array')
        screen = cv2.cvtColor(color_screen, cv2.COLOR_BGR2GRAY) if grayscale else color_screen
        # cv2.imshow('image', screen)
        return screen