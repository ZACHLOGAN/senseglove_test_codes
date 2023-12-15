import numpy as np
import csv
import random
import itertools


positions = np.arange(0,180)
#print(positions)
positions = list(positions)
sampled_positions = np.random.choice(positions, size = 15, replace = True)
sampled_positions = list(sampled_positions)
motor_output = [255*(item/180) for item in sampled_positions]

data = []
i = 0
while i <= len(sampled_positions)-1:
    values = [sampled_positions[i], motor_output[i]]
    data.append(values)
    i = i + 1

with open('network_values.csv', 'w', newline = '') as csvfile:
    writen = csv.writer(csvfile, delimiter = ',')
    writen.writerows(data)
