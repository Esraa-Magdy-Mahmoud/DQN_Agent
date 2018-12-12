from __future__ import print_function
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD , Adam
import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import tensorflow as tf
import json
# environment variables
state_size = (80,80,4)
action_size = 2
# exploration variables
eps_final = 0.0001
Membuffer_size = 50000
pretrain_buffer = 999999999
learning_rate = 0.00001
# Q Params
gamma = 0.9
Max_training_steps = 5000000
def DQN_Model(state_size,action_size,learning_rate):
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(80,80,4)))  #80*80*4
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))
   
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse',optimizer=sgd)
    return model

def preprocess(current_frame):
    current_frame = skimage.color.rgb2gray(current_frame)
    current_frame = skimage.transform.resize(current_frame,(80,80))
    current_frame = skimage.exposure.rescale_intensity(current_frame,out_range=(0,255))
    return current_frame

def new_action(eps,action,stacked_state,model):
    check = random.random()
    if check <= eps:
        print(" choosing random action !!")
        index = random.randrange(action_size)
        action[index] = 1
        return action,index
    else:
        print(" choosing action with Q !!")
        predicted_Q = model.predict(stacked_state)       
        max_Q = np.argmax(predicted_Q)
        index = max_Q
        action[max_Q] = 1
        return action,index

def play(model):
    #create Game env
    env = game.GameState()
    Memory_buffer = deque()
    ## intializing state
    action = np.array([1,0])
    state, reward, done = env.frame_step(action)
    state  = preprocess(state)
    stacked_state = np.stack((state, state, state, state), axis=2)
    stacked_state = stacked_state.reshape(1, stacked_state.shape[0], stacked_state.shape[1], stacked_state.shape[2])  

    eps_now = eps_final
    model.load_weights("./models/model.h5")
    sgd = SGD(lr=learning_rate)
    model.compile(loss='mse',optimizer=sgd)
    print ("Model load successfully")  
    step = 0
    while(True):
        loss = 0
        Q_DASH = 0
        index = 0
        reward = 0
        action = np.zeros(action_size)
        # choose an action
        action,index = new_action(eps_now,action,stacked_state,model)
        
        # take step forward into the env
        next_state, reward, done = env.frame_step(action)
        next_state = preprocess(next_state)

        next_state = next_state.reshape(1, next_state.shape[0], next_state.shape[1], 1) #1,64,64,1
        next_stacked_state = np.append(next_state, stacked_state[:, :, :, :3], axis=3)
        # Add the experience to the memory buffer
        Memory_buffer.append((stacked_state, index, reward, next_stacked_state, done))
        if len(Memory_buffer) > Membuffer_size:
            Memory_buffer.popleft()
        stacked_state = next_stacked_state
        step +=1
        print("STEP", step,\
             "/ ACTION", index, "/ REWARD", reward, \
            "/ Q_MAX " , np.max(Q_DASH),)
        print("finished an episode!")

def main():
    model = DQN_Model(state_size,action_size,learning_rate)
    play(model)
    

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    main()
        

