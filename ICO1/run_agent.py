from __future__ import print_function
import numpy as np
import sys
from collections import deque

sys.path.append('./bin/python')
import os.path
import vizdoom
from agent.doom_simulator import DoomSimulator
from agent.agent import Agent
from agent.trace import Trace
from agent.icolearning import Icolearning
import tensorflow as tf
import cv2


def main():
    ## Simulator
    simulator_args = {}
    simulator_args['config'] = 'config/config.cfg'
    simulator_args['resolution'] = (160, 120)
    simulator_args['frame_skip'] = 1
    simulator_args['color_mode'] = 'GRAY'
    simulator_args['game_args'] = "+name IntelAct +colorset 7"

    ## Agent
    agent_args = {}

    # preprocessing
    agent_args['preprocess_input_images'] = lambda x: x / 255. - 0.5
    agent_args['preprocess_input_measurements'] = lambda x: x / 100. - 0.5
    agent_args['num_future_steps'] = 6
    pred_scale_coeffs = np.expand_dims(
        (np.expand_dims(np.array([8., 40., 1.]), 1) * np.ones((1, agent_args['num_future_steps']))).flatten(), 0)
    agent_args['meas_for_net_init'] = range(3)
    agent_args['meas_for_manual_init'] = range(3, 16)

    # net parameters
    agent_args['conv_params'] = np.array([(16, 5, 4), (32, 3, 2), (64, 3, 2), (128, 3, 2)],
                                         dtype=[('out_channels', int), ('kernel', int), ('stride', int)])
    agent_args['fc_img_params'] = np.array([(128,)], dtype=[('out_dims', int)])
    agent_args['fc_meas_params'] = np.array([(128,), (128,), (128,)], dtype=[('out_dims', int)])
    agent_args['fc_joint_params'] = np.array([(256,), (256,), (-1,)], dtype=[('out_dims', int)])
    agent_args['target_dim'] = agent_args['num_future_steps'] * len(agent_args['meas_for_net_init'])

    # experiment arguments
    agent_args['test_objective_params'] = (np.array([5, 11, 17]), np.array([1., 1., 1.]))
    agent_args['history_length'] = 3
    agent_args['history_length_ico'] = 3

    print('starting simulator')
    simulator = DoomSimulator(simulator_args)

    print('started simulator')

    agent_args['discrete_controls'] = simulator.discrete_controls
    agent_args['continuous_controls'] = simulator.continuous_controls
    agent_args['state_imgs_shape'] = (
    agent_args['history_length'] * simulator.num_channels, simulator.resolution[1], simulator.resolution[0])

    agent_args['n_ffnet_hidden'] = np.array([50, 50])

    if 'meas_for_net_init' in agent_args:
        agent_args['meas_for_net'] = []
        for ns in range(agent_args['history_length']):
            agent_args['meas_for_net'] += [i + simulator.num_meas * ns for i in agent_args['meas_for_net_init']]
        agent_args['meas_for_net'] = np.array(agent_args['meas_for_net'])
    else:
        agent_args['meas_for_net'] = np.arange(agent_args['history_length'] * simulator.num_meas)
    if len(agent_args['meas_for_manual_init']) > 0:
        agent_args['meas_for_manual'] = np.array([i + simulator.num_meas * (agent_args['history_length'] - 1) for i in
                                                  agent_args[
                                                      'meas_for_manual_init']])  # current timestep is the last in the stack
    else:
        agent_args['meas_for_manual'] = []
    agent_args['state_meas_shape'] = (len(agent_args['meas_for_net']),)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    img_buffer = np.zeros(
        (agent_args['history_length'], simulator.num_channels, simulator.resolution[1], simulator.resolution[0]), dtype='uint8')
    img_buffer_ico = np.zeros(
        (agent_args['history_length_ico'], simulator.num_channels, simulator.resolution[1], simulator.resolution[0]), dtype='uint8')
    meas_buffer = np.zeros((agent_args['history_length'], simulator.num_meas))
    act_buffer = np.zeros((agent_args['history_length'], 7))
    act_buffer_ico = np.zeros((agent_args['history_length_ico'], 7))
    curr_step = 0
    old_step = -1
    term = False

    print ("state_meas_shape: ", meas_buffer.shape, " == ", agent_args['state_meas_shape'])
    print ("act_buffer_shape: ", act_buffer.shape)
    agent_args['n_ffnet_meas'] = len(np.ndarray.flatten(meas_buffer))
    agent_args['n_ffnet_act'] = len(np.ndarray.flatten(act_buffer))

    diff_y = 0
    diff_x = 0
    diff_z = 0
    diff_theta = 0
    iter = 1
    epoch = 200
    radialFlowLeft = 30
    radialFlowRight = 30
    radialFlowInertia = 0.4
    radialGain = 4.
    rotationGain = 100.
    errorThresh = 0
    updatePtsFreq = 50
    skipImage = 1
    skipImageICO = 5
    reflexGain = 1.

    # create masks for left and right visual fields - note that these only cover the upper half of the image
    # this is to help prevent the tracking getting confused by the floor pattern
    maskLeft = np.zeros([simulator_args['resolution'][1], simulator_args['resolution'][0]], np.uint8)
    maskLeft[simulator_args['resolution'][1]/2:, :simulator_args['resolution'][0]/2] = 1.
    maskRight = np.zeros([simulator_args['resolution'][1], simulator_args['resolution'][0]], np.uint8)
    maskRight[simulator_args['resolution'][1]/2:, simulator_args['resolution'][0]/2:] = 1.
    print (maskLeft)
    print (maskRight)

    userdoc = os.path.join(os.path.expanduser("~"), "Documents")

    # inputs: reflex + image + first 6 actions
    icoLeft = Icolearning(num_inputs= 1 + simulator_args['resolution'][0] * simulator_args['resolution'][1] + 7, num_filters=1)
    icoRight = Icolearning(num_inputs= 1 + simulator_args['resolution'][0] * simulator_args['resolution'][1] + 7, num_filters=1)

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=500, qualityLevel=0.03, minDistance=7, blockSize=7)
    imgCentre = np.array([simulator_args['resolution'][0] / 2, simulator_args['resolution'][1] /2])
    print ("Image centre: ", imgCentre)


    while not term:
        if curr_step < 100: #agent_args['history_length']:
            curr_act = np.zeros(7).tolist()
            img, meas, rwrd, term = simulator.step(curr_act)
            if curr_step == 0:
                p0Left = cv2.goodFeaturesToTrack(img, mask=maskLeft, **feature_params)
                p0Right = cv2.goodFeaturesToTrack(img, mask=maskRight, **feature_params)

            img_buffer[curr_step % agent_args['history_length']] = img
            meas_buffer[curr_step % agent_args['history_length']] = meas
            act_buffer[curr_step % agent_args['history_length']] = curr_act[:7]


        else:
            img1 = img_buffer[(curr_step-2) % agent_args['history_length'],0,:,:]
            img2 = img_buffer[(curr_step-1) % agent_args['history_length'],0,:,:]

            if(curr_step == 0 or curr_step % updatePtsFreq == agent_args['history_length']):
                print ("updating tracking points")
                p0Left = cv2.goodFeaturesToTrack(img, mask=maskLeft, **feature_params)
                p0Right = cv2.goodFeaturesToTrack(img, mask=maskRight, **feature_params)

            p1Left, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0Left, None, **lk_params)
            p1Right, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0Right, None, **lk_params)
            flowLeft = (p1Left - p0Left)[:,0,:]
            flowRight = (p1Right - p0Right)[:,0,:]
            radialFlowTmpLeft = 0
            radialFlowTmpRight = 0

            for i in range(0, len(p0Left)):
                radialFlowTmpLeft += ((p0Left[i,0,:] - imgCentre)).dot(flowLeft[i,:]) / float(len(p0Left))
            for i in range(0, len(p0Right)):
                radialFlowTmpRight += ((p0Right[i,0,:] - imgCentre)).dot(flowRight[i,:]) / float(len(p0Right))

            rotation = act_buffer[(curr_step - 1) % agent_args['history_length']][6]
            forward = act_buffer[(curr_step - 1) % agent_args['history_length']][3]
            # keep separate radial errors for left and right fields
            radialFlowLeft = radialFlowLeft + radialFlowInertia * (radialFlowTmpLeft - radialFlowLeft)
            radialFlowRight = radialFlowRight + radialFlowInertia * (radialFlowTmpRight - radialFlowRight)
            expectFlowLeft = radialGain * forward + rotationGain * rotation if rotation < 0. else 0.
            expectFlowRight = radialGain * forward - rotationGain * rotation if rotation > 0. else 0.
            flowErrorLeft = forward * (expectFlowLeft - radialFlowLeft) / (1. + rotationGain * np.abs(rotation))
            flowErrorRight = forward * (expectFlowRight - radialFlowRight) / (1. + rotationGain * np.abs(rotation))

            icoControlLeft = icoLeft.prediction(np.concatenate(
                ([(flowErrorLeft - errorThresh) if (flowErrorLeft - errorThresh) > 0. else 0. / reflexGain], np.ndarray.flatten(img), curr_act[:7])))
            icoControlRight = icoRight.prediction(np.concatenate(
                ([(flowErrorRight - errorThresh) if (flowErrorRight - errorThresh) > 0. else 0. / reflexGain], np.ndarray.flatten(img), curr_act[:7])))

#            print ("ICO input: ", [(flowErrorLeft - errorThresh) if (flowErrorLeft - errorThresh) > 0. else 0. / reflexGain], " : ",
#                   [(flowErrorRight - errorThresh) if (flowErrorRight - errorThresh) > 0. else 0. / reflexGain])

            print("** Expected ", expectFlowLeft, " ", expectFlowRight, " Actual: ", radialFlowLeft, " ", radialFlowRight, " err ", flowErrorLeft, " ", flowErrorRight, " ICOcontrol: ", icoControlLeft, " ", icoControlRight)

            diff_theta = .3 * min(icoControlRight - icoControlLeft, 30.)
            #            diff_z = -10. #* min(icoControl, 1.)

            curr_act = np.zeros(7).tolist()

            curr_act[0] = 0
            curr_act[1] = 0
            curr_act[2] = 0
            curr_act[3] = curr_act[3] + diff_z
            curr_act[3] = 10.
            curr_act[4] = 0
            curr_act[5] = 0

            curr_act[6] = curr_act[6] + diff_theta
#            curr_act[6] = 0.

            img, meas, rwrd, term = simulator.step(curr_act)
            if (not (meas is None)) and meas[0] > 30.:
                meas[0] = 30.

# Nnet to learn to predict next image - disabled at the moment
#            if (not (img is None)):

#                ag.act_ffnet(np.ndarray.flatten(state_imgs), np.ndarray.flatten(state_meas),
#                             np.array(np.ndarray.flatten(act_buffer), dtype='float64'), np.ndarray.flatten(ag.preprocess_input_images(img)))
#                diff_image = np.absolute(np.reshape(np.array(ag.ext_ffnet_output),
#                                                    [img.shape[0], img.shape[1]]) - ag.preprocess_input_images(img))
#                diff_image = np.absolute(ag.preprocess_input_images(img_buffer[(curr_step-1) % agent_args['history_length']] - ag.preprocess_input_images(img)))
#                diff_image = ag.preprocess_input_images(img)

#                diff_x = diff_x + inertia * (
#                (np.argmax(diff_image.sum(axis=0)) / float(diff_image.shape[1])) - 0.5 - diff_x)
#                diff_y = diff_x + inertia * (
#                (np.argmax(diff_image.sum(axis=1)) / float(diff_image.shape[0])) - 0.5 - diff_y)

#                print ("diff_x: ", diff_x, " diff_y: ", hack[6], "centre_x: ", np.argmax(diff_image.sum(axis=0)), "centre_y: ", np.argmax(diff_image.sum(axis=1)))

#                if (curr_step % epoch == 0):
#                    print("saving...")
#                    np.save(os.path.join('/home/paul', "hack"),
#                            np.reshape(np.array(ag.ext_ffnet_output), [img.shape[0], img.shape[1]]))
#                    np.save(os.path.join('/home/paul', "target"), ag.preprocess_input_images(img))
#                    np.save(os.path.join('/home/paul', "diff"), diff_image)

        if not term:
            img_buffer[curr_step % agent_args['history_length']] = img
            meas_buffer[curr_step % agent_args['history_length']] = meas
            act_buffer[curr_step % agent_args['history_length']] = curr_act[:7]

            if (curr_step - old_step) % skipImageICO == 0:
                img_buffer_ico[(curr_step // skipImage) % agent_args['history_length_ico']] = img
                act_buffer_ico[(curr_step // skipImage) % agent_args['history_length_ico']] = curr_act[:7]

            old_step = curr_step
            curr_step += 1

        if curr_step % epoch == 0:
            np.save('/home/paul/Dev/GameAI/vizdoom_cig2017/icolearner/ICO1/weights/icoLeft-' + str(curr_step), icoLeft.weights)
            np.save('/home/paul/Dev/GameAI/vizdoom_cig2017/icolearner/ICO1/weights/icoRight-' + str(curr_step), icoRight.weights)
            print ("saving weights... ")

    simulator.close_game()
    ag.save('/home/paul/Dev/GameAI/vizdoom_cig2017/icolearner/ICO1/checkpoints/' + 'hack-' + str(iter))


if __name__ == '__main__':
    main()