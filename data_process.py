import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from vehicle_data_gen_utils.utils import DataProcessor, ConfigJSON, Logger

TEST = 0
# TRAIN_DATADIR = '/workspace/data/tuner/fric3_rand_acc2_t02'
# TRAIN_DATADIR = '/home/lucerna/Documents/DATA/tuner_inn/track39'
TRAIN_DATADIR = './data/fric3_rand_f8'
if TEST:
    DATADIR = TRAIN_DATADIR + '_test/'
    # if this directory does not exist, create it
    try:
        os.mkdir(DATADIR)
    except OSError:
        print ("Creation of the directory %s failed" % DATADIR)
    else:
        print ("Successfully created the directory %s " % DATADIR)
else:
    DATADIR = TRAIN_DATADIR + '/'
TRAIN_SEGMENT = 2
TIME_INTERVAL = 0.2
SAVE_NAME = '_f5'

logger = Logger(DATADIR, SAVE_NAME)
logger.write_file(__file__)

# vlist = np.hstack([np.arange(0, 1, 0.1) + i for i in np.arange(5, 9)])
vlist = np.arange(8.0, 16.0, 1)
# flist = [0.5, 0.8, 1.1]
flist = [0.5]
dp = DataProcessor()
c = ConfigJSON()
all_friction_states = [] # length of flist: concatenate all data set over different speed command with this specific friction in flist
all_friction_control = []
selected_colums = [-1, 2, 5, 4] # steering angle, vx, yaw rate, vy. The original file has all the vehicle states, 7 columns
for ind, friction_ in enumerate(flist):
    total_states = []
    total_controls = []
    for vel in vlist:
        filename = 'states_f{}_v{}.npy'.format(int(friction_*10),
                                                       int(np.rint(vel*100)))
        controls_filename = 'controls_f{}_v{}.npy'.format(int(friction_*10), 
                                                                  int(np.rint(vel*100)))
        
        states = np.load(DATADIR + filename)
        states = states[:,:,selected_colums]
        controls = np.load(DATADIR + controls_filename)
        total_states.append(states[:, :])
        total_controls.append(controls[:, :])

    all_friction_states.append(np.vstack(total_states))
    all_friction_control.append(np.vstack(total_controls))
                    ## x3 = steering angle of front wheels
                    ## x4 = velocity in x-direction
                    ## x6 = yaw rate
                    ## x11 = velocity in y-direction
        #   vehicle_state = np.array([x,  # x
        #                       y,  # y
        #                       vx,  # vx
        #                       yaw_angle,  # yaw angle
        #                       vy,  # vy
        #                       yaw_rate,  # yaw rate
        #                       steering  # steering angle
        #                     ])     
     
## normalization
normalization_param = []
for ind in range(4):
    _, param = dp.data_normalize(np.vstack(np.vstack(all_friction_states))[:, ind])
    normalization_param.append(param)

dynamics = []
for ind, friction_ in enumerate(flist):
    states_fric = all_friction_states[ind] # (320=# of segment,210=length of segment,4=# of states)
    # controls_fric = all_friction_control[ind]
    for segment_ind in range(states_fric.shape[0]): # segment_ind:0~319
        states = states_fric[segment_ind] # (210,4)
        # controls = controls_fric[segment_ind]
        states = np.vstack([states[i:i+2][None, :] for i in range(0, len(states)-2+1, 1)]) # (105(rangeを2個飛ばしでやったら)/209(range1個飛ばし),2,4)
        dynamics.append((states[:, 1, 1:] - states[:, 0, 1:]) / TIME_INTERVAL) # vx, yaw rate, vy. steering angleは除いた
        
dynamics = np.vstack(dynamics) # (320*209, 3)
for ind in range(3): #dynamicsはvx, yaw rate, vyの三列
    _, param = dp.data_normalize(dynamics[:, ind])
    normalization_param.append(param)
    
for ind in range(2): #controlは二列
    _, param = dp.data_normalize(np.vstack(np.vstack(all_friction_control))[:, ind])
    normalization_param.append(param)
print(normalization_param)
    
c.d['normalization_param'] = normalization_param
c.save_file(DATADIR + 'config' + SAVE_NAME + '.json')

if TEST:
    c = ConfigJSON()
    c.load_file(TRAIN_DATADIR + '/config' + SAVE_NAME + '.json')
    normalization_param = c.d['normalization_param']

train_states_fric = []
train_controls_fric = []
train_dynamics_fric = []
train_labels_fric = []
# normalization_param[0] = [2, -1]

# How's this for loop different from the for loop in l.75?

for ind, friction_ in enumerate(flist):
    states_fric = all_friction_states[ind] # (320=# of segment,210=length of segment,4=# of states)
    controls_fric = all_friction_control[ind] # (320,210,2)
    
    train_states = []
    train_controls = []
    train_dynamics = []
    train_labels = []
    
    for segment_ind in range(states_fric.shape[0]): # segment_ind:0~319
    # for segment_ind in range(1):
        states = states_fric[segment_ind] # (210,4)
        controls = controls_fric[segment_ind] # (210,2)
        
        # states = np.vstack([states[i:i+TRAIN_SEGMENT][None, :] for i in range(1, len(states)-TRAIN_SEGMENT+1, TRAIN_SEGMENT)]) # (104,2,4)
        # controls = np.vstack([controls[i:i+TRAIN_SEGMENT][None, :] for i in range(1, len(controls)-TRAIN_SEGMENT+1, TRAIN_SEGMENT)]) # (104,2,2)
        states = np.vstack([states[i:i+TRAIN_SEGMENT][None, :] for i in range(1, len(states)-TRAIN_SEGMENT+1, 1)]) # (104,2,4)
        controls = np.vstack([controls[i:i+TRAIN_SEGMENT][None, :] for i in range(1, len(controls)-TRAIN_SEGMENT+1, 1)]) # (104,2,2)
        dynamics = (states[:, 1:, 1:] - states[:, :-1, 1:]) / TIME_INTERVAL
        label = [ind] * dynamics.shape[0]
        
        train_states.append(states)
        train_controls.append(controls)
        train_dynamics.append(dynamics)
        train_labels.append(label)
            
    train_states_fric.append(np.vstack(train_states))
    train_controls_fric.append(np.vstack(train_controls)) # (320*104,2,4)をappend
    train_dynamics_fric.append(np.vstack(train_dynamics))
    train_labels_fric.append(np.hstack(train_labels))
    
train_states_fric = np.array(train_states_fric)
train_controls_fric = np.array(train_controls_fric)
train_dynamics_fric = np.array(train_dynamics_fric)
train_labels_fric = np.array(train_labels_fric)
    
print('train_states', train_states_fric.shape) # (1(=# of friction), 33280, 2, 4), 1個飛ばしなら(1, 66560(=320*208), 2, 4)
print('train_controls_fric', train_controls_fric.shape) # (1, 33280, 2, 2)
print('train_dynamics_fric', train_dynamics_fric.shape) # (1, 33280, 1, 3)
print('train_labels', train_labels_fric.shape) # (1, 33280)



np.savez(DATADIR + 'train_data' + SAVE_NAME, 
         train_states=train_states_fric, 
         train_controls=train_controls_fric, 
         train_dynamics=train_dynamics_fric,
         train_labels=train_labels_fric)
