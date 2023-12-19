#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension, Float32MultiArray, Float32, Float64, String
import numpy as np
from math import pi
import torch
import torch.nn as nn
from network import FeedForwardNN

class ESP32_Commands:

	def __init__(self):
		
		#subscriber for updating the PPO Neural Network
		self.Update_PPO_NN = rospy.Subscriber("PPO_Network", String, self.update_learn_NN)
		#subscriber for recieveing the current position and direction of the user
		self.position_data = rospy.Subscriber("ESP32_Data", FLoat64MultiArray, self.send_commands)
		
		# set dimensions of action and observation space
		self.obs_dim = 1
		self.act_dim = 1
		
		#initialize the baseline policy NN
		Baseline_model = FeedForwardNN(obs_dim, act_dim)
		#initialize the PPO policy NN
		PPO_model = FeedForwardNN(obs_dim, act_dim)
		
	#subscriber callback that updates the NN with the changes created by the PPO algorithm
	def update_learn_NN(self, weights):
		new_weight = eval(weights)
	
	#subscriber callback that runs the entire generation and publishing of the ESP32 motor commands	
	def send_commands(self, movements):
	
