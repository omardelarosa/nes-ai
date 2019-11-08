from rom_wrapper import ROMWrapper
from nes_py.wrappers import JoypadSpace
from nes_py._image_viewer import ImageViewer
from nes_py.app.play_human import play_human
from nes_py.app.cli import _get_args
import time

import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

args = _get_args()

# required arg
rom_path = args.rom


# create the environment
env = ROMWrapper(rom_path)

SHOULD_RENDER = True
FLAT_STATE_SIZE = 2048
GAMMA = 0.2
log_interval = 100
num_steps_per_episode = 2250
reward_threshold = 1500.0
# DQR STUFF


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(FLAT_STATE_SIZE, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    flat_state = state.flatten()
    # print("state_shape: ", state.shape, flat_state.shape)
    state = torch.from_numpy(flat_state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + GAMMA * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


# MAIN FUNCTIONS
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
        ['down', 'A'],
        ['down', 'B'],
        ['up', 'A'],
        ['up', 'B'],
        ['up'],
        ['down'],
        ['A'],
        ['B']
    ]

    env = JoypadSpace(env, actions)

    env.reset()

    action = 0
    # start = time.time()

    running_reward = 10
    for i_episode in count(1):
        print("Episode: ", i_episode)
        _, ep_reward = env.reset(), 0
        state = np.zeros(FLAT_STATE_SIZE)  # Initial state
        for t in range(1, num_steps_per_episode):  # Don't infinite loop while learning
            action = select_action(state)
            _, reward, done, info = env.step(action)
            state = info['ram']
            if SHOULD_RENDER:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        print("Running reward: ", running_reward)
        if running_reward > reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

    # # play_human
    # for t in range(0, steps):
    #     # get the mapping of keyboard keys to actions in the environment
    #     if hasattr(env, 'get_keys_to_action'):
    #         keys_to_action = env.get_keys_to_action()
    #     elif hasattr(env.unwrapped, 'get_keys_to_action'):
    #         keys_to_action = env.unwrapped.get_keys_to_action()
    #     else:
    #         raise ValueError('env has no get_keys_to_action method')

    #     # # change action every 6 frames
    #     if t % 6 == 0:
    #         action = env.action_space.sample()

    #         # after 500 timesteps, stop pressing start button
    #         if t > 500:
    #             while action == 0:
    #                 action = env.action_space.sample()

    #     observation, reward, done, info = env.step(action)
    #     print("---------------------------t: ", t)
    #     print("action space: ", action, env.action_space)
    #     print("obs: ", observation.shape)
    #     print("reward: ", reward)
    #     print("info: ", info)
    #     # runs game at about 60fps
    #     time.sleep(0.016667)
    #     env.render()

    # end = time.time()
    # env.close()
    # print("time: ", (end - start), " seconds  for ", steps, "steps")


if args.mode == 'human':
    print("Playing as human")
    play_human_custom(env)
else:
    print("Playing with random custom agent")
    play_random_custom(env, args.steps)
