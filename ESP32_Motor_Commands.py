#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension, Float32MultiArray, Float32, Float64
import numpy as np
from math import pi


class Motion_Commands:
    def __init__(self):
        
        self.DT = 1./200. #refresh rate for data collection and publishing in seconds
        self.counter = 0 #timer counter in seconds
        self.max_pos = 180 #maximum possible fingertip position in degrees
        self.trial = 0 #index for tracking when a trial is active

    #variables that can be editted by ppo
        #vibration pulse duration (50 ms)
        self.activation = 0.05 #seconds
        #vibration pulse delay (every 1/8 of a second)
        self.delay = 0.125 #seconds
        #index marking that tactor was activated
        self.active = 0 # 0 means not active, 1 means active
        
        #scaler for adjusting the scale of vibration amplitude
        self.sensitivity = 1 #number between 0-1

        #scaler for determining the amount of deviation from 0 twist for direction
        self.alpha = 0.5 #scaler from 0-1

        #variables for position data to be saved to
        self.index_current = 0 #most recent position of index finger
        self.index_prior = 0 #prior position of index finger
        self.index_direction = 0 #direction of index finger motion

        self.thumb_current = 0 #most recent position of thumb
        self.thumb_prior = 0 #prior position of thumb
        self.thumb_lean = 0 #current tilt of the thumb
        self.thumb_direction = 0 #direction of thumb motion

        self.ring_current = 0 #most recent position of ring finger
        self.ring_prior = 0 #prior position of ring finger
        self.ring_direction = 0 #direction of ring finger motion

        self.ring_current = 0 #most recent position of middle finger
        self.ring_prior = 0 #prior position of middle finger
        self.ring_direction = 0 #direction of middle finger motion

        #zero twist positions for servos, all given in degrees
        self.t_twist = 90
        self.i_twist = 90
        self.m_twist = 90
        self.r_twist = 90

        #variables for containing vibration motor values
        self.m1 = 0
        self.m2 = 0
        self.m3 = 0
        self.m4 = 0
        #variable for containing twist servo values
        self.s1 = 90
        self.s2 = 90
        self.s3 = 90
        self.s4 = 90
        
        #variable for handling the position values for esp32
        self.esp32_pmotor_values = Float32MultiArray()
        self.esp32_pmotor_values.layout = MultiArrayLayout()
        self.esp32_pmotor_values.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
        self.esp32_pmotor_values.layout.dim[0].size = 1
        self.esp32_pmotor_values.layout.dim[1].size = 2
        self.esp32_pmotor_values.data = [0, 90]

        #variable for handling the force values for esp32
        self.esp32_fmotor_values = Float32MultiArray()
        self.esp32_fmotor_values.layout = MultiArrayLayout()
        self.esp32_fmotor_values.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
        self.esp32_fmotor_values.layout.dim[0].size = 1
        self.esp32_fmotor_values.layout.dim[1].size = 3
        self.esp32_fmotor_values.data = [0, 0, 0]

        #publisher for esp32 serial connection
        self.esp32_values = rospy.Publisher("esp32_values", Float32MultiArray, queue_size = 5)
        #subscriber for recieving the feedback position data from SenseGlove
        self.pos_fingers = rospy.Subscriber("finger_position", Float64MultiArray, self.pos_fingertip)
        #subscriber for handling fingertip forces
        self.force_fingers = rospy.Subscriber("finger_forces", Float64MultiArray, self.force_fingertip)

        #subscriber for tracking trial state
        self.trial_progress = rospy.Subscriber("trial_progress", Float64, self.trial_index)
        #subsrciber for promximal policy updates

    def direction_comp(self):
        self.index_direction = self.index_current - self.index_prior
        self.thumb_direction = self.thumb_current - self.thumb_prior
        self.ring_direction = self.ring_current - self.ring_prior
        self.middle_direction = self.middle_current - self.middle_prior

        #check index finger direction
        if (self.index_direction < 0):
            self.i_twist = 90 - self.alpha #rotate twist mechanism 90 degress counter-clockwise 

        else:
            self.i_twist = 90 + self.alpha #rotate twist mechanism 90 degrees clockwise

        #check thumb direction
        if (self.thumb_direction < 0):
            self.t_twist = 90 - self.alpha #rotate twist mechanism 90 degress counter-clockwise 

        else:
            self.t_twist = 90 + self.alpha #rotate twist mechanism 90 degrees clockwise

        #check middle finger direction
        if (self.middle_direction < 0):
            self.m_twist = 90 - (90*self.alpha) #rotate twist mechanism 90 degress counter-clockwise 

        else:
            self.m_twist = 90 + (90*self.alpha) #rotate twist mechanism 90 degrees clockwise

        #check ring finger direction
        if (self.ring_direction < 0):
            self.r_twist = 90 - (90*self.alpha) #rotate twist mechanism 90 degress counter-clockwise 

        else:
            self.r_twist = 90 + (90*self.alpha) #rotate twist mechanism 90 degrees clockwise

        twist = [self.i_twist, self.t_twist, self.m_twist, self.r_twist]

        return twist

    #subcriber callback function for parsing out fingertip position from senseglove data
    def pos_fingertip(self, data):

        self.position = [self.index_current, self.thumb_current, self.middle_current, self.ring_current]
        
        return self.position

    #subscriber callback function for parsing out fingertip force from sensor data
    def force_fingertip(self, data):

        return;

    #function for generating the needed motor commands for the esp32 with snaptics
    def motor_commands(self):
        if (self.pos_trial == 1): #if 1 then doing position feedback trials
            #obtain direction information for twist system
            twist = direction_comp()
            #save twist values to new variable
            self.s1 = twist[0]
            
            #save current position as the last position for next direction calculation
            self.index_prior = self.index_current
            self.thumb_prior = self.thumb_current
            self.middle_prior = self.middle_current
            self.ring_prior = self.ring_current


            #generate motor commands for vibration using 255 pwm scale based on fingertip position ratio
            self.m1 = 255*self.sensitivity*(self.index_current/self.max_pos)
            motor_values = [self.m1, self.s1]
            self.esp32_pmotor_values.data = motor_values
            

        else: #otherwise we are doing force feedback trials


        
    def motor_activation(self, event):

        #during trial and tactor is to be active publish non-zero motor commands to esp32
        if(self.trail == 1 and self.active == 1):

            #publish all commands to esp32
            self.esp32_values.publish(self.esp32_pmotor_values)
            #self.esp32_values.publish(self.esp32_fmotor_values)
            
            self.counter = self.counter + self.DT #update the timing counter value
            if (self.counter == self.activation): #check if counter has reached pulse length
                self.counter = 0 #set counter back to 0
                self.active = 0 #set active value to 0 to turn off tactor
                
        #during trial and tactor is not active publish zero valued motor commands to esp32
        elif(self.trial == 1 and self.active == 0):
            #set all motor values to be 0 positions
            self.esp32_pmotor_values.data = [0, 90]
            #self.esp32_fmotor_values.data = [0, 0, 0]

            #publish all commands to esp32
            self.esp32_values.publish(self.esp32_pmotor_values)
            #self.esp32_values.publish(self.esp32_fmotor_values)

            
            self.counter = self.counter + self.DT #update the timing counter value
            if(self.counter == self.delay): #check if timing counter has reached delay length
                self.counter = 0 #set counter time back to 0
                self.active = 1 #set active value to 1 to turn tactor back on




            
