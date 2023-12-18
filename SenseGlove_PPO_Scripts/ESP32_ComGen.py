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

        #variables for storing feedback and position data
        self.feedback = 0
        self.current_position = 0
        self.velocity = 0

        #variables for initial servo twist values
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
        
        #variables for defining vibration pulse length
        #vibration pulse duration (50 ms)
        self.activation = 0.05 #seconds
        #vibration pulse delay (every 1/8 of a second)
        self.delay = 0.125 #seconds
        #index marking that tactor was activated
        self.active = 0 # 0 means not active, 1 means active

        #variables for esp publisher
        self.esp32_motor_values = Float32MultiArray()
        self.esp32_motor_values.layout = MultiArrayLayout()
        self.esp32_motor_values.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
        self.esp32_motor_values.layout.dim[0].size = 1
        self.esp32_motor_values.layout.dim[1].size = 2
        self.esp32_motor_values.data = [0, 90]
        
        #publisher for esp32 motor commands
        self.esp_commands = rospy.Publisher("ESP32_Commands", Float32MultiArray, queue_size = 5)
        #subscriber to trial condition
        self.trial_state = rospy.Subscriber("trial_progress", Float64, self.trial_activation)
        #subscriber to position, velocity and feedback data
        self.e32 = rospy.Subscriber("ESP32_Data", Float64MultiArray, self.pos_state)

        #subscriber callback to get trial state
        def trial_activation(self, data):
            self.trial = data.data
            
        #subsriber callback to get feedback condition, user position, user velocity, user error
        def pos_state(self, data):
            self.feedback = data.data[0]
            self.current_position = data.data[1]
            self.velocity = data.data[2]
            error = data.data[3]

        def direction_comp(self):

            #check index finger direction
            if (self.velocity < 0):
                self.i_twist = 90 - self.alpha #rotate twist mechanism 90 degress counter-clockwise 

            else:
                self.i_twist = 90 + self.alpha #rotate twist mechanism 90 degrees clockwise
                

        #function for generating the needed motor commands for the esp32 with snaptics
        def motor_commands(self):
            if (self.pos_trial == 1): #if 1 then doing position feedback trials
                #obtain direction information for twist system
                twist = direction_comp()
                #save twist values to new variable
                self.s1 = twist[0]

                #generate motor commands for vibration using 255 pwm scale based on fingertip position ratio
                self.m1 = 255*self.sensitivity*(self.current_position/self.max_pos)
                motor_values = [self.m1, self.s1]
                self.esp32_motor_values.data = motor_values

        def motor_activation(self, event):
            if(self.feedback == 1 or self.feedback == 2):
                #during trial and tactor is to be active publish non-zero motor commands to esp32
                if(self.trail == 1 and self.active == 1):

                    #publish all commands to esp32
                    self.esp32_values.publish(self.esp32_motor_values)
                    
                    self.counter = self.counter + self.DT #update the timing counter value
                    if (self.counter == self.activation): #check if counter has reached pulse length
                        self.counter = 0 #set counter back to 0
                        self.active = 0 #set active value to 0 to turn off tactor
                        
                #during trial and tactor is not active publish zero valued motor commands to esp32
                elif(self.trial == 1 and self.active == 0):
                    #set all motor values to be 0 positions
                    self.esp32_pmotor_values.data = [0, 90]

                    #publish all commands to esp32
                    self.esp32_values.publish(self.esp32_motor_values)

                    
                    self.counter = self.counter + self.DT #update the timing counter value
                    if(self.counter == self.delay): #check if timing counter has reached delay length
                        self.counter = 0 #set counter time back to 0
                        self.active = 1 #set active value to 1 to turn tactor back on
            
#main function of the node
def main():
    def main():
    #initialize the ros node under name Manager
    rospy.init_node('Manager', log_level = rospy.INFO)
    try:
        #initialize the class variable for running the code
        commander = Motion_Commands()
    except rospy.ROSInterruptException:
        pass

#run the ros node
if __name__=='__main__':
    main()

    
