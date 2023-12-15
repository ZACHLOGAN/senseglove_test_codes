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

        self.feedback = Float64()
        self.feedback.data = 0.0

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
        
        #variable for saving and transfering current session state to RVIZ
        #data order is [session, desired_position, user_position]
        self.rviz_data = Float64MultiArray()
        self.rviz_data.layout = MultiArrayLayout()
        self.rviz_data.layout.dim = [MultiArrayDimension()]
        self.rviz_data.layout.dim[0].size = 1
        self.rviz_data.layout.dim[0].stride = 3
        self.rviz_data.data = [0, 0, 0]

        #variable for saving position and velocity data to be sent to esp32 and/or ppo plus feedback state, plus error
        self.data_32 = Float64MultiArray()
        self.data_32.layout = MultiArrayLayout()
        self.data_32.layout.dim = [MultiArrayDimension()]
        self.data_32.layout.dim[0].size = 1
        self.data_32.layout.dim[0].stride = 4
        self.data_32.data = [0, 0, 0, 0]
        
        #publisher for all desired data to be set for recording script 
        self.record_data = rospy.Publisher("Data_Record", Float64MultiArray, queue_size = 5)		

        #publish current trial state
        self.trial_progress = rospy.Publisher("trial_progress", Float64, queue_size = 5)

        
        #publish position and velocity ESP32 Command GEN and/or PPO
        self.esp32 = rospy.Publisher("ESP32_Data", FLoat64MultiArray(), queue_size = 5)

        #publish current session state to RVIZ
        self.session_state = rospy.Publisher("RVIZ_Session", Float64MultiArray, queue_size = 5)
        
        #subsrciber for user fingertip position
        self.user_pos = rospy.Subscriber("finger_position", Float64MultiArray, self.data_logger)

        
    #function for generating particpant trial setup
    def pos_trial_generation(self):
        #angular fingertip positions to be tested (0 is fully open, 180 is fully closed)
        positions = [0, 45, 90, 135, 180]
        repeats_base = 10
        repeats_adapt = 20
        repeats_post = 10
        #feedback types, 0 = vision only, 1 = vision and baseline policy, 2 = vision and updating policy
        feedback_method = [0, 1, 2]
        practice_trials = 10
        baseline_trials = len(positions)*repeats_base#number of baseline session trials
        adaptation_trials = len(positions)*repeats_adapt #number of motor adaptation session trials
        post_trials = len(positions)*repeats_post #number of post-adaptation session trials

        c = random.sample(feedback_method, 1)
        base = []
        adapt = []
        post = []
        trials = []
        k = 1
        j = 0
        while (j < len(positions)-1):
            k = 1
            while (k <= repeats_base):
                tips = [c[0], 1, positions[j]]
                base.append(tips)
                k = k + 1
            k = 1
            while(k <= repeats_adapt):
                tips = [c[0], 2, positions[j]]
                adapt.append(tips)
                k = k + 1
            k = 1
            while(k <= repeats_post):
                tips = [c[0], 3, positions[j]]
                post.append(tips)
                k = k + 1
            j = j + 1
        random.shuffle(base)
        random.shuffle(adapt)
        random.shuffle(post)

        k = 0
        while (k<= len(base)-1):
            trials.append(base[k])
            k  = k + 1

        k = 0
        while (k<= len(adapt)-1):
            trials.append(adapt[k])
            k  = k + 1

        k = 0
        while (k<= len(post)-1):
            trials.append(post[k])
            k  = k + 1
        return trials    

    #obtain senseglove position data
    def data_logger(self, data):
        self.position = data.data
    #compute the direction of user motion and their error
    def compute_velocity_error(self):
        self.velocity = (self.position-self.prior_position)/(self.DT)
        self.error = self.position - self.current_trial[2]
        self.data_32.data = [self.velocity, self.error]
    
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
            #concatenate 
            self.logged.data = [self.current_trial[0], self.current_trial[1], self.current_trial[2], self.counter, self.position, self.velocity, self.error]
            
            #if a trial is active then publish data to respective topics
            if(self.session_index == 1):
                self.session_state.publish(self.rviz_data)
                self.record_data.publish(self.logged)
                self.counter = self.counter + self.DT
                                           
            elif(self.session_index == 2):
                self.session_state.publish(self.rviz_data)
                self.record_data.publish(self.logged)
                self.counter = self.counter + self.DT
                
            elif(self.session_index == 3):
                self.session_state.publish(self.rviz_data)
                self.record_data.publish(self.logged)
                self.counter = self.counter + self.DT

    def train(self, 

def main():
    #initialize the ros node under name Manager
    rospy.init_node('Manager', anonymous = False, log_level = rospy.INFO, disable_signals = False)

    #initialize the class variable for running the code
    manager = Trial_Controller()

    #create or load participant data
    data = manager.pos_trial_generation()
    
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
            i = i + 1
            
    except rospy.ROSInterruptException:
        pass
    rospy.signal_shutdown("Testing is Complete")
    rospy.spin()

#run the ros node
if __name__ == '__main__':
    main()
            
            
