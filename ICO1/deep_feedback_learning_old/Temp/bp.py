#!/usr/bin/python3

from __future__ import print_function
from vizdoom import *
import threading
import math

import sys
from time import sleep
from matplotlib import pyplot as plt

sys.path.append('../../deep_feedback_learning')

import numpy as np
import cv2
import deep_feedback_learning

# Create DoomGame instance. It will run the game and communicate with you.
game = DoomGame()

# Now it's time for configuration!
# load_config could be used to load configuration instead of doing it here with code.
# If load_config is used in-code configuration will also work - most recent changes will add to previous ones.
# game.load_config("../../scenarios/basic.cfg")

# Sets path to additional resources wad file which is basically your scenario wad.
# If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
game.set_doom_scenario_path("./basic.wad")

# Sets map to start (scenario .wad files can contain many maps).
game.set_doom_map("map01")

# Sets resolution. Default is 320X240
game.set_screen_resolution(ScreenResolution.RES_640X480)

# create masks for left and right visual fields - note that these only cover the upper half of the image
# this is to help prevent the tracking getting confused by the floor pattern
width = 640
widthNet = 320
height = 480
heightNet = 240

# Sets the screen buffer format. Not used here but now you can change it. Defalut is CRCGCB.
game.set_screen_format(ScreenFormat.RGB24)

# Enables depth buffer.
game.set_depth_buffer_enabled(True)

# Enables labeling of in game objects labeling.
game.set_labels_buffer_enabled(True)

# Enables buffer with top down map of the current episode/level.
game.set_automap_buffer_enabled(True)

# Sets other rendering options
game.set_render_hud(False)
game.set_render_minimal_hud(False)  # If hud is enabled
game.set_render_crosshair(True)
game.set_render_weapon(False)
game.set_render_decals(False)
game.set_render_particles(False)
game.set_render_effects_sprites(False)
game.set_render_messages(False)
game.set_render_corpses(False)

# Adds buttons that will be allowed. 
# game.add_available_button(Button.MOVE_LEFT)
# game.add_available_button(Button.MOVE_RIGHT)
game.add_available_button(Button.MOVE_LEFT_RIGHT_DELTA, 50)
game.add_available_button(Button.ATTACK)
game.add_available_button(Button.TURN_LEFT_RIGHT_DELTA)

# Adds game variables that will be included in state.
game.add_available_game_variable(GameVariable.AMMO2)

# Causes episodes to finish after 200 tics (actions)
game.set_episode_timeout(500)

# Makes episodes start after 10 tics (~after raising the weapon)
game.set_episode_start_time(10)

# Makes the window appear (turned on by default)
game.set_window_visible(True)

# Turns on the sound. (turned off by default)
game.set_sound_enabled(True)

# Sets the livin reward (for each move) to -1
game.set_living_reward(-1)

# Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
game.set_mode(Mode.PLAYER)

game.set_seed(1)

# Enables engine output to console.
#game.set_console_enabled(True)

nFiltersInput = 0
nFiltersHidden = 0
minT = 3
maxT = 30
nHidden0 = 4
nHidden1 = 2
net = deep_feedback_learning.DeepFeedbackLearning(widthNet*heightNet,[nHidden0*nHidden0], 1, nFiltersInput, nFiltersHidden, minT,maxT)
net.seedRandom(1)
net.getLayer(0).setConvolution(widthNet,heightNet)
#net.getLayer(1).setConvolution(nHidden0,nHidden0)
net.initWeights(0.1,1,deep_feedback_learning.Neuron.MAX_OUTPUT_RANDOM);
net.setLearningRate(0.01)
net.setMomentum(0.9)
net.setAlgorithm(deep_feedback_learning.DeepFeedbackLearning.backprop);
# net.getLayer(0).setInputNorm2ZeroMean(128,256)
net.getLayer(0).setLearningRate(1E-3)
net.getLayer(1).setLearningRate(1E-3)
#net.getLayer(2).setLearningRate(1E-3)
#net.getLayer(2).setLearningRate(1E-2)
#net.getLayer(1).setNormaliseWeights(True)
#net.getLayer(2).setNormaliseWeights(True)
net.setUseDerivative(0)
net.setBias(1.)

# Initialize the game. Further configuration won't take any effect from now on.
game.init()

# Run this many episodes
episodes = 1000

# Sets time that will pause the engine after each action (in seconds)
# Without this everything would go too fast for you to keep track of what's happening.
zsleep_time = 1.0 / DEFAULT_TICRATE # = 0.028

delta2 = 0
dontshoot = 1
deltaZeroCtr = 1

inp = np.zeros(widthNet*heightNet)

sharpen = np.array((
	[0, 1, 0],
	[1, 4, 1],
	[0, 1, 0]), dtype="int")

edge = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")






plt.ion()
plt.show()
ln1 = [False,False]
ln2 = [False,False,False,False]

def getWeights2D(neuron, method='wt'):
    n_neurons = net.getLayer(0).getNneurons()
    n_inputs = net.getLayer(0).getNeuron(neuron).getNinputs()
    weights = np.zeros(n_inputs)
    for i in range(n_inputs):
        if net.getLayer(0).getNeuron(neuron).getMask(i):
            if(method == 'wt'):
                weights[i] = net.getLayer(0).getNeuron(neuron).getAvgWeight(i)
            elif(method == 'ch'):
                weights[i] = net.getLayer(0).getNeuron(neuron).getAvgWeightCh(i)

        else:
            weights[i] = np.nan
    return weights.reshape(heightNet,widthNet)

def getWeights1D(layer,neuron):
    n_neurons = net.getLayer(layer).getNneurons()
    n_inputs = net.getLayer(layer).getNeuron(neuron).getNinputs()
    weights = np.zeros(n_inputs)
    for i in range(n_inputs):
        weights[i] = net.getLayer(layer).getNeuron(neuron).getAvgWeight(i)
    return weights

def plotWeights():
    global ln1
    global ln2

    while True:

        if ln1[0]:
            ln1[0].remove()
        plt.figure(1)
        w1 = getWeights2D(0, 'wt')
        for i in range(1,net.getLayer(0).getNneurons()):
            w2 = getWeights2D(i, 'wt')
            w1 = np.where(np.isnan(w2),w1,w2)
        ln1[0] = plt.imshow(w1,cmap='gray')
#        plt.draw()
#        plt.pause(0.1)

        if ln1[1]:
            ln1[1].remove()
        plt.figure(2)
        w1 = getWeights2D(0, 'ch')
        for i in range(1,net.getLayer(0).getNneurons()):
            w2 = getWeights2D(i, 'ch')
            w1 = np.where(np.isnan(w2),w1,w2)
        ln1[1] = plt.imshow(w1,cmap='gray')
        plt.draw()
        plt.pause(0.1)

        for j in range(1,net.getNumHidLayers()+1):
            if ln2[j]:
                ln2[j].remove()
            plt.figure(j+2)
            w1 = np.zeros( (net.getLayer(j).getNneurons(),net.getLayer(j).getNeuron(0).getNinputs()) )
            for i in range(0,net.getLayer(j).getNneurons()):
                w1[i,:] = getWeights1D(j,i)
            ln2[j] = plt.imshow(w1,cmap='gray')
            plt.draw()
            plt.pause(0.1)


t1 = threading.Thread(target=plotWeights)
t1.start()



for i in range(episodes):
#    print("Episode #" + str(i + 1))

    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    game.new_episode()

    while not game.is_episode_finished():

        # Gets the state
        state = game.get_state()

        # Which consists of:
        n = state.number
        vars = state.game_variables
        screen_buf = state.screen_buffer
        depth_buf = state.depth_buffer
        labels_buf = state.labels_buffer
        automap_buf = state.automap_buffer
        labels = state.labels

        midlinex = int(width/2);
        midliney = int(height*0.75);
        crcb = screen_buf
        screen_left = screen_buf[100:midliney,0:midlinex-1,2]
        screen_right = screen_buf[100:midliney,midlinex+1:(width-1),2]
        screen_left = cv2.filter2D(screen_left, -1, sharpen)
        screen_right = cv2.filter2D(screen_right, -1, sharpen)
#        cv2.imwrite('/tmp/left.png',screen_left)
#        cv2.imwrite('/tmp/right.png',screen_right)
        lavg = np.average(screen_left)
        ravg = np.average(screen_right)
        delta = (lavg - ravg)*15
        dd = delta - delta2
        delta2 = delta
#        print(delta)

        # Makes a random action and get remember reward.
        shoot = 0
        if (dontshoot > 1) :
            dontshoot = dontshoot - 1
        else :
            if (abs(dd) < 10) :
                shoot = 1
                dontshoot = 60
                deltaZeroCtr = 4

        if deltaZeroCtr>0:
            deltaZeroCtr = deltaZeroCtr - 1
            delta = 0

        if (delta>30.):
            delta=30.
        if (delta<-30.):
            delta=-30.

        blue = cv2.resize(crcb, (widthNet,heightNet))
        blue = blue[:,:,2]
#        blue = cv2.filter2D(blue, -1, edge)
        err = np.linspace(delta,delta,nHidden0*nHidden0)

        net.setAlgorithm(deep_feedback_learning.DeepFeedbackLearning.backprop)
        # net.getLayer(0).setInputNorm2ZeroMean(128,256)
#        net.getLayer(0).setLearningRate(1E-6)
#        net.getLayer(1).setLearningRate(1E-4)
        # net.getLayer(2).setLearningRate(1E-2)
#        net.getLayer(1).setNormaliseWeights(True)
        # net.getLayer(2).setNormaliseWeights(True)
#        net.setUseDerivative(0)

        net.doStep(blue.flatten()/512-0.5, [0.])
        output = net.getOutput(0)*40.
        neterr = delta - output

        net.doStep(blue.flatten()/512-0.5, [neterr/40.])

        #weightsplot.set_xdata(np.append(weightsplot.get_xdata(),n))
        #weightsplot.set_ydata(np.append(weightsplot.get_ydata(),net.getLayer(0).getWeightDistanceFromInitialWeights()))

        output = net.getOutput(0)*30.
        print(delta,output,
              net.getLayer(0).getWeightDistanceFromInitialWeights(),"\t",
              net.getLayer(1).getWeightDistanceFromInitialWeights(),"\t",
              " Err: ", net.getOutputLayer().getNeuron(0).getError())
#       action[0] is translating left/right; action[2] is rotating/aiming
#        action = [ delta+output , shoot, 0. ]
        action = [ 0., shoot, (delta)*0.1 ]
        r = game.make_action(action)

        if (i%20 == 0):
            net.saveModel()
#        input('')

#        if sleep_time > 0:
#            sleep(sleep_time)

    # Check how the episode went.
#    print("Episode finished.")
#    print("Total reward:", game.get_total_reward())
#    print("************************")
    sleep(1)

# It will be done automatically anyway but sometimes you need to do it in the middle of the program...
game.close()
