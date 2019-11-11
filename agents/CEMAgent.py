from envs.cem_rom_wrapper import CEMROMWrapper
from nes_py.wrappers import JoypadSpace
from nes_py._image_viewer import ImageViewer
from nes_py.app.play_human import play_human
from nes_py.app.cli import _get_args
import time

import gym
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CEMAgent(nn.Module):
	def __init__(self, args, h_size = 50):
		super(CEMAgent, self).__init__()
		print("Cross Entropy Method")

		rom_path = args.rom

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

		self.env = CEMROMWrapper(rom_path)
		self.env = JoypadSpace(self.env, actions)
		self.s_size = self.env.observation_space.shape[0]*self.env.observation_space.shape[1]
		self.h_size = h_size
		self.a_size = self.env.action_space.n
		print("s size: ", self.s_size)
		print("h size: ", self.h_size)
		print("a size: ", self.a_size)
		self.fc1 = nn.Linear(self.s_size, self.h_size)
		print(self.fc1)
		self.fc2 = nn.Linear(self.h_size, self.a_size)
		print(self.fc2)
		#self.fc3 = nn.linear(self.a_size)

		self.scores = self.cem()

	def set_weights(self,weights):
		s_size = self.s_size
		h_size = self.h_size
		a_size = self.a_size
		# separate the weights for each layer
		fc1_end = ( s_size*h_size ) + h_size
		fc1_W = torch.from_numpy(weights[:s_size*h_size].reshape(s_size, h_size))
		fc1_b = torch.from_numpy(weights[s_size*h_size:fc1_end])
		fc2_W = torch.from_numpy(weights[fc1_end:fc1_end+(h_size*a_size)].reshape(h_size, a_size))
		fc2_b = torch.from_numpy(weights[fc1_end+(h_size*a_size):])
		# set the weights for each layer
		self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
		self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
		self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
		self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))

	def get_weights_dim(self):
		return (self.s_size + 1)* self.h_size + (self.h_size + 1)*self.a_size

	def forward(self, x):
		#print("x: ",x.shape)
		x = torch.tanh(self.fc1(x.reshape(-1)))
		x = torch.tanh(self.fc2(x))
		return x.cpu().data

	def evaluate(self, weights, gamma = 1.0, max_t = 5000):
		self.set_weights(weights)
		episode_return = 0.0
		state = self.env.reset()
		turns_since_change_in_reward = 0
		last_reward = 0
		for t in range(max_t):
			#print(state.shape)
			scaler = StandardScaler()
			state = scaler.fit_transform(np.dot(state.copy(),[0.299, 0.587, 0.114]))
			state = torch.from_numpy(state).float().to(device)
			#state = np.dot(torch.from_numpy(state.copy()).float().to(device),[0.299, 0.587, 0.114])
			#print(state.shape)
			output = (self.forward(state))
			print("Output Vector: ",output)
			print(self.fc1.weight.data)
			best_val = -100
			if t < 500:
				best_ind = 0
				start_comp = 1
			else:
				best_ind = 1
				start_comp = 2
			
			for i in range(start_comp,self.a_size):
				#if output[i] > output[best_ind] and abs(output[i]-output[best_ind]) > 0.1:
				if output[i] > output[best_ind]:
					best_ind = i
					best_val = output[i]
				
			action = best_ind

			if(turns_since_change_in_reward == 10):
				action = self.env.action_space.sample()
				turns_since_change_in_reward =0

			print("Action Chose: ",action)
			state, reward, done, _ = self.env.step(action)
			if reward == last_reward:
				turns_since_change_in_reward +=1
			last_reward = reward
			self.env.render()
			print("Reward: ",reward)
			episode_return += reward * math.pow(gamma,t)
			if done:
				break
		return episode_return

	def cem(self,n_iterations = 500, max_t = 5000, gamma = 1.0, print_every = 10, pop_size = 50, elite_frac = 0.2, sigma = 0.5):
		n_elite=int(pop_size*elite_frac)
		scores_deque = deque(maxlen=100)
		scores = []
		best_weight = sigma*np.random.randn(self.get_weights_dim())

		for i_iteration in range(1, n_iterations+1):
			print("iteration ", i_iteration)
			weights_pop = [best_weight + (sigma*np.random.randn(self.get_weights_dim())) for i in range(pop_size)]
			rewards = np.array([self.evaluate(weights, gamma, max_t) for weights in weights_pop])

			elite_idxs = rewards.argsort()[-n_elite:]
			elite_weights = [weights_pop[i] for i in elite_idxs]
			best_weight = np.array(elite_weights).mean(axis=0)
			reward = self.evaluate(best_weight, gamma=1.0)
			scores_deque.append(reward)
			scores.append(reward)

			torch.save(self.state_dict(), 'checkpoint.pth')

			if i_iteration % print_every == 0:
				print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))

			if np.mean(scores_deque)>=90.0:
				print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
				break

		return scores


