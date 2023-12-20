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
		#subscriber for checking if a trial is in progress
		self.trial_state = rospy.Subscriber("trial_progress", Float64, self.trial_activation)
		
		#publisher for sending out the pwm and servo positions for esp32
		self.ESPER32 = rospy.Publisher("ESP32_Commands", Float32MultiArray, queue_size = 5)
		
		# set dimensions of action and observation space
		self.obs_dim = 1
		self.act_dim = 1
		
		#initialize the baseline policy NN
		Baseline_model = FeedForwardNN(obs_dim, act_dim)
		#load baseline model weights
		Baseline_model.load_state_dict(torch.load('my_network.pth'))
		#set Baseline model to evaluation mode
		Baseline_model.eval()
		
		#initialize the PPO policy NN
		PPO_model = FeedForwardNN(obs_dim, act_dim)
		#intialize baseline weights
		PPO_model.load_stat_dict(torch.load('my_network.pth'))
		#set PPO model to evaluation mode
		PPO_model.eval()
		
		#initialization values for motor commands
		self.serv_neutral = 90.0 # in degrees
		self.motor_base = 0.0 #as pwm signal
		self.trial_index = 0.0 #index to check if trial is running or not
		
		#initialize variable for publishing the esp32 motor commands
		self.commands = Float32MultiArray()
		self.commands.layout = MultiArrayLayout()
		self.commands.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
		self.commands.layout.dim[0].size = 1
		self.commands.layout.dim[1].size = 2
		self.commands.data = [self.motor_base, self.serv_neutral]
		
	#subscriber callback to check if trial is in progress
	def trial_activation(self, data):
		self.trial_index = data.data
	
	#subscriber callback that updates the NN with the changes created by the PPO algorithm
	def update_learn_NN(self, weights):
		#convert string message of state_dict to dictionary message
		new_state_dict = eval(weights)
		#update PPO model state dictionary
		PPO_model.load_state_dict(torch.load(new_state_dict))
		#ensure PPO nn is set back to eval mode
		PPO_model.eval()
	
	#subscriber callback that runs the entire generation and publishing of the ESP32 motor commands	
	def send_commands(self, movements):
		if(self.trial_index == 1):
			#seperate the position and velocity into their own variables
			position = movements.data[0]
			velocity = movements.data[1]
			session = movements.data[2]
			feedback = movements.data[3]
			
			#check if velocity is positive or negative for direction and set twist amount
			if (velocity > 0):
				servo_twist = self.serv_neutral + 30
			elif (velocity < 0):
				servo_twist = self.serv_neutral - 30
			else:
				servo_twist = self.serv_neutral
			
			#use the neural networks to compute the pwm motor command based on the current position
			#use if statements to check the feedback type to use and the session
			if (session == 1 and (feedback == 1 or feedback == 2)):
				#use baseline NN regardless of feedback type
				pwm_value = Baseline_model(position)
				
			elif (feedback == 1 and (session == 2 or session == 3)):
				#use baseline model for other sessions for those in baseline group
				pwm_value = Baseline_model(position)
				
			elif(feedback == 2 and (session == 2 or session == 3)):
				#use ppo model for other sessions for those in RL group
				pwm_value = PPO_model(position)
				
			self.commands.data = [pwm_value, servo_twist]
			self.ESPER32.publish(self.commands)
			
			
		else:
			#set vibration motor pwm to 0 and servo to neutral position
			self.commands.data = [self.motor_base, self.serv_neutral]
			#publish the ESP32 data for the device to get
			self.ESPER32.publish(self.commands)
			
#main loop for running the ROS Node			
def main():
	#initialize the command generator node
	rospy.init_node("ESP32_Command_Generator", log_level = rospy.INFO)
	try:
		generator = ESP32_Commands()
	except rospy.ROSInterruptException:
		pass
	rospy.spin()
	
#callout for launching the code
if __name__=='__main__':
	main()
