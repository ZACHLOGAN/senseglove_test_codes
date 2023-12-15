#gymnasium enviornment for prosthetic experiment
import gymnasium as gym
from gym import spaces
import numpy as np
#import pygame

class ProsMotorFeed(gym.ENV):
    def __init__(self):
        #define feedback limits and parameters
        self.max_pwm = 255
        self.min_pwm = 0
        self.max_position = 180
        self.min_position = 0

        
        low_obs = np.array([self.min_position, self.min_position], dtype = np.float32)
        high_obs = np.array([self.max_position, self.max_position], dtype = np.float32)
        #define action space
        self.action_space = spaces.Box(low = self.min_pwm, high = self.max_pwm, shape = (1,), dtype = np.float32)
        #define observation space
        self.observation_space = spaces.Box(low = low_obs, high = high_obs, dtype = np.float32)

    #step the environment forward in time
    def step(self, u):
        current = self
        #define system state
        current, desired = self.state
        #action forced to be in limits
        u = np.clip(u, self.min_pwn, self.max_pwm)[0]

        #reward function
        if (current - desired

    def pull_current(self)
        current = Current_Position()
        return current

    def pull_desired
