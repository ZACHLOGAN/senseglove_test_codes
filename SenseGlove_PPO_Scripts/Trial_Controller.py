#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension, Int16MultiArray
import numpy as np
from math import pi
import csv
import random
import itertools


#imports needed to run the ppo system
import torch


class Trial_Controller:
    def __init__(self):

        #refresh rate for data collection and publishing
        self.DT = 1./200.
        #index for marking the current session during experiment for data publishing, 0 = start, 1 = baseline, 2 = motor adaptation, 3 = post-adaptation, 4 = end
        self.session_index = 0
        #trial length
        self.endtime = 2.0 # in seconds
        #variable for user finger position
        self.position = 0.0
        #variable for denoting that a trial is in progress
        self.trial = Float64()
        self.trial.data = 0.0
        #variable for transfering trial data to data logger [feedback, session, desired_position]
        self.current_trial = [0, 0, 0]
        #variable for tracking trial length
        self.counter = 0
	
	#count number of data values collected from the system
	self.data_index = 1.0
	
	#value is that 0 is vision, 1 is baseline only, 2 is RL learned
        self.feedback = Float64()
        self.feedback.data = 0.0
	
	#running ppo variable
	self.tell_ppo = Int16()
	self.tell_ppo.data = 0
	self.is_ppo_done = 0
	
        self.position = 0
        self.prior_position = 0
        self.velocity = 0
        self.error = 0
        
        #Variable for storing recorded trial data
        #data is ordered [feedback_condition, session_type, trials_desired_position, trial_time, user_position, user_velocity, position_error]
        self.logged = Float64MultiArray()
        self.logged.layout = MultiArrayLayout()
        self.logged.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
        self.logged.layout.dim[0].size = 1
        self.logged.layout.dim[1].size = 7
        self.logged.data = [0, 0, 0, 0, 0, 0, 0]
        
        #varible for transfering data for ppo system to use 
        #data is ordered as [data_index, desired_position, actual_position]
        self.train_data = Float64MultiArray()
        self.train_data.layout = MultiArrayLayout()
        self.train_data.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
        self.train_data.layout.dim[0].size = 1
        self.train_data.layout.dim[1].size = 2
        self.train_data.data = [0, 0, 0]
        
        #variable for saving and transfering current session state to RVIZ
        #data order is [session, desired_position, user_position]
        self.rviz_data = Float64MultiArray()
        self.rviz_data.layout = MultiArrayLayout()
        self.rviz_data.layout.dim = [MultiArrayDimension()]
        self.rviz_data.layout.dim[0].size = 1
        self.rviz_data.layout.dim[0].stride = 3
        self.rviz_data.data = [0, 0, 0]

        #variable for saving position and velocity data to be sent to esp32
        #data order is [user position, user velocity, session, feedback type]
        self.data_32 = Float64MultiArray()
        self.data_32.layout = MultiArrayLayout()
        self.data_32.layout.dim = [MultiArrayDimension()]
        self.data_32.layout.dim[0].size = 1
        self.data_32.layout.dim[0].stride = 4
        self.data_32.data = [0, 0, 0, 0]
        
        
        #publisher for running the PPO Data
        self.Run_PPO = rospy.Publisher("Run_PPO", Int16, queue_size = 5)
        
        #publisher for all desired data to be set for recording script 
        self.record_data = rospy.Publisher("Data_Record", Float64MultiArray, queue_size = 5)		

        #publish current trial state
        self.trial_progress = rospy.Publisher("trial_progress", Float64, queue_size = 5)

        #publisher for data needed for ppo system to store
        self.ppo_publisher = rospy.Publisher("Recall_Data", Float64MultiArray, queue_size = 5)
        
        #publish position and velocity ESP32 Command GEN and/or PPO
        self.esp32 = rospy.Publisher("ESP32_Data", FLoat64MultiArray, queue_size = 5)

        #publish current session state to RVIZ
        self.session_state = rospy.Publisher("RVIZ_Session", Float64MultiArray, queue_size = 5)
        
        #subsrciber for user fingertip position
        self.user_pos = rospy.Subscriber("finger_position", Float64MultiArray, self.data_logger)
        
        #subscriber for checking if the ppo is done training
        self.check_done = rospy.Subscriber("Done_Check", Int16, self.update_done_check)

    #obtain senseglove position data
    def data_logger(self, data):
        self.position = data.data
        
    #subscriber callback to check if ppo is done
    def update_done_check(self, data):
    	self.is_ppo_done = data.data
    	
    #compute the direction of user motion and their error
    def compute_velocity_error(self):
        self.velocity = (self.position-self.prior_position)/(self.DT)
        self.error = self.position - self.current_trial[2]
        self.data_32.data = [self.position, self.velocity]
    
    #function for collecting, concatenating, and publishing experiment data to final record topic
    def data_record(self):
        #publish trial condition to motor_command node to force motors to be off when no trial is running
        self.trial_progress.publish(self.trial)
        
        if(self.trial.data == 1):
            #publish session data to rviz
            self.session_state.publish(self.rviz_data)
            self.compute_velocity_error()
            self.prior_position = self.position
            
            #publish the velocity and error data
            self.esp32.publish(self.data_32)
            #concatenate data into one variable for publishing to data record
            self.logged.data = [self.current_trial[0], self.current_trial[1], self.current_trial[2], self.counter, self.position, self.velocity, self.error]
            #concatenate data into one variable for publishing to RL system
            self.train_data.data = [self.data_index, self.current_trial[2], self.position]
            
            #if a trial is active then publish data to respective topics
            if(self.session_index == 1):
                self.session_state.publish(self.rviz_data)
                self.record_data.publish(self.logged)
                self.counter = self.counter + self.DT
                                           
            elif(self.session_index == 2):
                self.session_state.publish(self.rviz_data)
                self.record_data.publish(self.logged)
                #if RL is active publish to the RL learn variable for memory storing
                if (self.current_trial[0] == 2):
                	self.ppo_publisher.publish(self.train_data)
                	self.data_index = self.data_index + 1
                self.counter = self.counter + self.DT
                
            elif(self.session_index == 3):
                self.session_state.publish(self.rviz_data)
                self.record_data.publish(self.logged)
                self.counter = self.counter + self.DT

def main():
    #initialize the ros node under name Manager
    rospy.init_node('Manager', anonymous = False, log_level = rospy.INFO, disable_signals = False)

    #initialize the class variable for running the code
    manager = Trial_Controller()

    #create or load participant data
    data = manager.pos_trial_generation()
    #data = np.genfromtext("")
    
    #index for tracking overall trial progress, counts all session trials as time limits and recorded data is the same and is tracked in the data generation part
    i = 0
    try:
        while not rospy.is_shutdown():
            manager.trial.data = 0
            #check to see if all data has been completed    
            if(i > len(data)-1):
                rospy.loginfo("All Trials have Been Ran")
                rospy.signal_shutdown("Testing is Complete")
                rospy.spin()
            
            #ask tester for if the testing will continue
            move_forward = input("Continue Testing Y/N")

            #if testing is chosen to stop for any reason, shut down the ros node properly and finalize all publishing
            if(move_forward == 'n' or move_forward == 'no' or move_forward == 'N' or move_forward == 'No' or move_forward == 'NO'):
                rospy.log_info("Tester or Participant have chosen to stop test")
                rospy.signal_shutdown("Testing will end early")
                rospy.spin()

            #pull generated trial data out for use
            manager.current_trial = [data[i][0], data[i][1], data[i][2]]
            manager.feedback.data = data[i][0]

            manger.feedback_state.publish(manager.feedback)
            manger.feedback_state.publish(manager.feedback)
            
            #while loop to hold back the steping of trials till user has a chance to complete it
            manager.trial.data = 1
            manager.counter = 0
            while(manager.counter <= manager.endtime):
                a = 20

            manager.trial.data = 0
            manager.counter = 0
            manager.is_ppo_done = 0
            if (manager.current_trial[0] == 2 and (manager.session_index == 2 or manager.session_index == 3)):
            	manager.Run_PPO.publish(manager.tell_ppo)
            	while (manager.is_ppo_done == 0)
            		a = 20
            		if (manager.is_ppo_done == 1):
            			break;
            
            i = i + 1
            
    except rospy.ROSInterruptException:
        pass
    rospy.signal_shutdown("Testing is Complete")
    rospy.spin()

#run the ros node
if __name__ == '__main__':
    main()
