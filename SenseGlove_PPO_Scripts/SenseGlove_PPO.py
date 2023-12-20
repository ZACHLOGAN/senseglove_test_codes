#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension, String
import random
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from network import FeedForwardNN

class Sense_PPO:
	"""
	PPO class used to train our haptic feedback algorithm using SenseGlove Data. Will be directly called in Trial_Controller node.
	"""
	def __init__(self, **hyperparameters)
		#subscriber for recall the data collected during the trials
		self.recall_data = rospy.Subscriber("Recall_Data", Float64MultiArray, self.store_data)
		#subscriber for running the training algorithm
		self.Run_PPO = rospy.Subscriber("Run_PPO", Int16, self.Agent)
		#publisher for sending the updated neural network weights
		self.network_updater = rospy.Publisher("PPO_Network", String, queue_size = 5)
		#publisher to tell the main loop training is complete
		self.done_check = rospy.Publisher("Done_Check", Int16, queue_size = 5)
		
		#flag for checking if ppo is done
		self.done_flag = Int16()
		self.done_flag.data = 0
		
		#initialize the variable for the storing the data
		self.obsv_stored = []
		
		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)

		# set dimensions of action and observation space
		self.obs_dim = 1
		self.act_dim = 1

		# Initialize actor and critic networks using premade network script in Pytorch
		self.actor = FeedForwardNN(self.obs_dim, self.act_dim) # ALG STEP 1
		self.critic = FeedForwardNN(self.obs_dim, 1)

		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var)
		
		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}
		
	
	def PPOGloveReset(self):
		self.obsv_stored = []
	
	def store_data(self, obsv):
		#store all data put into the call in one large variable to use for batch generation and ppo training
		self.obsv_stored.append(obsv)
		
	def SenseGloveReset(self):
		#this function behaves as a within-script replacement for the OpenAI Gym env.reset() function
		#sample the first data point from the stored obsv data
		first_point = random.choice(self.obsv_stored)
		return first_point
		
	def SenseGloveStep(self, u, obsv):
		#this function behaves as a within-script replacement for the OpenAI Gym/Gymnasium env.step() function
		#using action generated by the actor NN compute the rewards based upon the observations and clip the results between the minimum and maximum values possible
		dpos, cpos = obsv
		
		#action clip function
		u = np.clip(,self.min_pwm, self.max_pwm)
		
		#compute the position error, computed on a 0 to 1 scale for ease of use
		error = abs(cpos - dpos)/self.max_pos
		#compute the rewards obtained by the system by using a range of allowed errors
		if (error > 0.20):
			rewards = -5
		elif (error > 0.10 and error < 0.20):
			rewards = -1
		elif (error == 0.10):
			rewards = 0
		elif(error < 0.10 and error > 0.05):
			rewards = 1
		elif(error < 0.10):
			rewards = 5
			
		#clip observations
		cpos = np.clip(cpos, self.min_pos, self.max_pos)
		dpos = np.clip(dpos, self.min_pos, self.max_pos)
		
		#determine when done flag gets triggered
		return obsv, rewards, done
		
		
		
		
	def generate_batches(self): #currently still has environment calls, convert to only use sampling of large data array.
		#generate the batches used by the ppo training algorithm
		"""
			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
		# Batch data. For more details, check function header.
		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_rews = []
		batch_rtgs = []
		batch_lens = []

		# Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
		ep_rews = []

		t = 0 # Keeps track of how many timesteps we've run so far this batch

		# Keep simulating until we've run more than or equal to specified timesteps per batch
		while t < self.timesteps_per_batch:
			ep_rews = [] # rewards collected per episode

			#reset is to take the first sample for the batch 
			obs = self.SenseGloveReset()
			done = False

			# Run an episode for a maximum of max_timesteps_per_episode timesteps
			for ep_t in range(self.max_timesteps_per_episode):
				t += 1 # Increment timesteps ran this batch so far

				# Track observations in this batch
				batch_obs.append(obs)

				# Calculate action and make a step in the env. 
				# Note that rew is short for reward.
				action, log_prob = self.get_action(obs)
				obs, rew, done, _, _ = self.env.step(action)

				# Track recent reward, action, and action log probability
				ep_rews.append(rew)
				batch_acts.append(action)
				batch_log_probs.append(log_prob)

				# If the environment tells us the episode is terminated, break
				if done:
					break

			# Track episodic lengths and rewards
			batch_lens.append(ep_t + 1)
			batch_rews.append(ep_rews)

		# Reshape data as tensors in the shape specified in function description, before returning
		batch_obs = torch.tensor(batch_obs, dtype=torch.float)
		batch_acts = torch.tensor(batch_acts, dtype=torch.float)
		batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
		batch_rtgs = self.compute_rtgs(batch_rews)                                                              # ALG STEP 4

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
	
	
	def Agent(self, data):
		#ppo training algorithm
		while t_so_far < total_timesteps:                                                                       # ALG STEP 2
			#create batches of data to train ppo on
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.generate_batches()                     # ALG STEP 3

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)

			# Increment the number of iterations
			i_so_far += 1

			# Logging timesteps so far and iterations so far
			self.logger['t_so_far'] = t_so_far
			self.logger['i_so_far'] = i_so_far

			# Calculate advantage at k-th iteration
			V, _ = self.evaluate(batch_obs, batch_acts)
			A_k = batch_rtgs - V.detach()                                                                       # ALG STEP 5

			# One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
			# isn't theoretically necessary, but in practice it decreases the variance of 
			# our advantages and makes convergence much more stable and faster. I added this because
			# solving some environments was too unstable without it.
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			# This is the loop where we update our network for some n epochs
			for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
				# Calculate V_phi and pi_theta(a_t | s_t)
				V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				# NOTE: we just subtract the logs, which is the same as
				# dividing the values and then canceling the log with e^log.
				# For why we use log probabilities instead of actual probabilities,
				# here's a great explanation: 
				# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
				# TL;DR makes gradient ascent easier behind the scenes.
				ratios = torch.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				# Calculate actor and critic losses.
				# NOTE: we take the negative min of the surrogate losses because we're trying to maximize
				# the performance function, but Adam minimizes the loss. So minimizing the negative
				# performance function maximizes it.
				actor_loss = (-torch.min(surr1, surr2)).mean()
				critic_loss = nn.MSELoss()(V, batch_rtgs)

				# Calculate gradients and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()

				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach())
				
		#extract state dictionary from actor model and convert to string
		trained_state_dictionary = self.actor.state_dict()
		trained_state_dictionary_str = str(trained_state_dictionary)
		#publish trained model to string message for use in command evaluation of ESP32
		self.network_updater.publish(trained_state_dictionary_str)
		
		#tell system that ppo is done
		self.done_flag.data = 1
		self.done_check.publish(self.done_flag)
		

	#computes the rewards to go of each timestep contained in the batch
	def compute_rtgs(self, batch_rews):
		"""
			Compute the Reward-To-Go of each timestep in a batch given the rewards.

			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num timesteps per episode)
		batch_rtgs = []

		# Iterate through each episode
		for ep_rews in reversed(batch_rews):

			discounted_reward = 0 # The discounted reward so far

			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return (think about why it would be harder starting from the beginning)
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

		return batch_rtgs
	
	#function for computing the mean action of the actor NN
	def get_action(self, obs):
		# Query the actor network for a mean action
		mean = self.actor(obs)

		# Create a distribution with the mean action and std from the covariance matrix above.
		# For more information on how this distribution works, check out Andrew Ng's lecture on it:
		# https://www.youtube.com/watch?v=JjB58InuTqM
		dist = MultivariateNormal(mean, self.cov_mat)

		# Sample an action from the distribution
		action = dist.sample()

		# Calculate the log probability for that action
		log_prob = dist.log_prob(action)

		# Return the sampled action and the log probability of that action in our distribution
		return action.detach().numpy(), log_prob.detach()
	
	#function for computing the value for each batch based on the critic NN
	def evaluate(self, batch_obs, batch_acts):	
		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
		V = self.critic(batch_obs).squeeze()

		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()
		mean = self.actor(batch_obs)
		dist = MultivariateNormal(mean, self.cov_mat)
		log_probs = dist.log_prob(batch_acts)

		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
		return V, log_probs
		
	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.

			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
		self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
		self.lr = 0.005                                 # Learning rate of actor optimizer
		self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
		# Miscellaneous parameters
		#self.render = True                              # If we should render during rollout
		#self.render_every_i = 10                        # Only render every n iterations
		#self.save_freq = 10                             # How often we save in number of iterations
		self.seed = None                                # Sets the seed of our program, used for reproducibility of results
		
		#hyperparameters denoting input and observation limits (need to be floats)
		self.max_pwm = 255.0
		self.min_pwm = 0.0
		self.max_pos = 180.0
		self.min_pos = 0.0
		
		"""
		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			exec('self.' + param + ' = ' + str(val))
		"""
		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)
			# Set the seed 
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")

#main loop for running the code as a ROS node		
def main():
	rospy.init_node("PPO_Training_Alg", log_level = rospy.INFO)
	try:
		train_my_algorithm = Sense_PPO()
	except rospy.ROSInterruptException:
		pass
	rospy.spin()

#code for launching the code
if __name__=='__main__':
	main()
