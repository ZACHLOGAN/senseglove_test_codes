import sys
import torch

#rename the from names to be my names
from ppo import PPO
from network import FeedForwardNN
from eval_policy import eval_policy


def train(env, hyperparameters, actor_model, critic_model):

	model.learn(total_timesteps=100_000)
