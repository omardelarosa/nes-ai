from rom_wrapper import ROMWrapper
from nes_py.wrappers import JoypadSpace
from nes_py._image_viewer import ImageViewer
from nes_py.app.play_human import play_human
from nes_py.app.cli import _get_args
import time

import numpy as np
import math
import random
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.distributions import Categorical

args = _get_args()

# required arg
rom_path = args.rom

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device: ", device)

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

# create the environment
env = ROMWrapper(rom_path)

screen_height = 240
screen_width = 256
steps_done = 0

DATA_PATH = "data/dqr_model.pt"
SHOULD_LOAD_STATE = True
SHOULD_RENDER = True
SHOULD_TRAIN = True
FLAT_STATE_SIZE = 2048
GAMMA = 0.2
log_interval = 100
num_steps_per_episode = 2250
reward_threshold = 1500.0

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

# DQR STUFF


class Policy(nn.Module):
    def __init__(self, h, w, outputs):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    # cart_location = get_cart_location(screen_width)
    # if cart_location < view_width // 2:
    #     slice_range = slice(view_width)
    # elif cart_location > (screen_width - view_width // 2):
    #     slice_range = slice(-view_width, None)
    # else:
    #     slice_range = slice(cart_location - view_width // 2,
    #                         cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    # screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


# MAIN FUNCTIONS
def play_human_custom(env):
    """
    re-using default nes-py play human function
    https://github.com/Kautenja/nes-py/blob/1dda1ad37a84e3ca67dfbebd7cc0c2d8e4cf2489/nes_py/app/play_human.py

    """
    play_human(env)


def play_random_custom(env, steps):
    _NOP = 0

    env = JoypadSpace(env, actions)

    env.reset()

    action = 0
    start = time.time()

    if SHOULD_TRAIN:

        init_screen = get_screen()
        _, _, screen_height, screen_width = init_screen.shape

        # INIT Neural Network
        policy = Policy(screen_height, screen_width, len(actions))

        if SHOULD_LOAD_STATE:
            print("Loading model from: ", DATA_PATH)
            policy.load_state_dict(torch.load(DATA_PATH))

        optimizer = optim.Adam(policy.parameters(), lr=1e-2)
        eps = np.finfo(np.float32).eps.item()

        # Helper functions
        def select_action(state):
            global steps_done
            sample = random.random()
            eps_threshold = reward_threshold
            # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            #     math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    return policy(state).max(1)[1].view(1, 1)
            else:
                return torch.tensor([[random.randrange(len(actions))]], device=device, dtype=torch.long)

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
            print("POLICY LOSS: ", policy_loss)
            # policy_loss = torch.cat(policy_loss).sum()
            # policy_loss.backward()
            optimizer.step()
            torch.save(policy.state_dict(), DATA_PATH)
            del policy.rewards[:]
            del policy.saved_log_probs[:]

        running_reward = 10
        for i_episode in count(1):
            print("Episode: ", i_episode)
            state, ep_reward = env.reset(), 0
            for t in range(1, num_steps_per_episode):  # Don't infinite loop while learning
                action = select_action(state).data.cpu().numpy()[0][0]
                # print("ACTION:", action)
                state, reward, done, info = env.step(action)
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
    else:
        # PLAY RANDOMLY
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
            print("---------------------------t: ", t)
            print("action space: ", action, env.action_space)
            print("obs: ", observation.shape)
            print("reward: ", reward)
            print("info: ", info)
            # runs game at about 60fps
            time.sleep(0.016667)
            env.render()

    end = time.time()
    env.close()
    print("time: ", (end - start), " seconds  for ", steps, "steps")


if args.mode == 'human':
    print("Playing as human")
    play_human_custom(env)
else:
    print("Playing with random custom agent")
    play_random_custom(env, args.steps)
