import random
import itertools
import numpy as np
import math

def rot_matrix(deg):
    theta = deg*(math.pi/180)
    c = np.cos(theta)
    s = np.sin(theta)
    return [c, s]

#angular fingertip positions to be tested (0 is fully open, 180 is fully closed)
positions = [0, 45, 90, 135, 180]
repeats_base = 5
repeats_adapt = 10
repeats_post = 5
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
#print(trials)

xy = np.array([2,5])
r = rot_matrix(30)
r_rot = np.array([[r[0],-r[1]],[r[1],r[0]]])
rot_xy = r_rot.dot(xy)
print(rot_xy)
