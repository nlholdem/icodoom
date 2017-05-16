from __future__ import print_function
import numpy as np
import sys
from collections import deque

sys.path.append('./bin/python')
import os.path
import vizdoom
from agent.doom_simulator import DoomSimulator
from agent.agent import Agent
from agent.firfilter import Filterbank
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
    agent_args['postprocess_predictions'] = lambda x: x * pred_scale_coeffs
    agent_args['discrete_controls_manual'] = range(6, 12)
    agent_args['meas_for_net_init'] = range(3)
    agent_args['meas_for_manual_init'] = range(3, 16)
    agent_args['opposite_button_pairs'] = [(0, 1), (2, 3)]

    # net parameters
    agent_args['conv_params'] = np.array([(16, 5, 4), (32, 3, 2), (64, 3, 2), (128, 3, 2)],
                                         dtype=[('out_channels', int), ('kernel', int), ('stride', int)])
    agent_args['fc_img_params'] = np.array([(128,)], dtype=[('out_dims', int)])
    agent_args['fc_meas_params'] = np.array([(128,), (128,), (128,)], dtype=[('out_dims', int)])
    agent_args['fc_joint_params'] = np.array([(256,), (256,), (-1,)], dtype=[('out_dims', int)])
    agent_args['target_dim'] = agent_args['num_future_steps'] * len(agent_args['meas_for_net_init'])

    # efference copy

    # experiment arguments
    agent_args['test_objective_params'] = (np.array([5, 11, 17]), np.array([1., 1., 1.]))
    agent_args['history_length'] = 3
    agent_args['test_checkpoint'] = 'model'

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
    meas_buffer = np.zeros((agent_args['history_length'], simulator.num_meas))
    act_buffer = np.zeros((agent_args['history_length'], 6))
    curr_step = 0
    term = False

    print ("state_meas_shape: ", meas_buffer.shape, " == ", agent_args['state_meas_shape'])
    print ("act_buffer_shape: ", act_buffer.shape)
    agent_args['n_ffnet_meas'] = len(np.ndarray.flatten(meas_buffer))
    agent_args['n_ffnet_act'] = len(np.ndarray.flatten(act_buffer))


    ag = Agent(sess, agent_args)
    ag.load('./checkpoints')
    filterBank = Filterbank(3)

    acts_to_replace = [a + b + d + e for a in [[0, 0], [1, 1]] for b in [[0, 0], [1, 1]] for d in [[0]] for e in
                       [[0], [1]]]
    print(acts_to_replace)
    replacement_act = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    # MOVE_FORWARD   MOVE_BACKWARD   TURN_LEFT   TURN_RIGHT  ATTACK  SPEED   SELECT_WEAPON2  SELECT_WEAPON3  SELECT_WEAPON4  SELECT_WEAPON5  SELECT_WEAPON6  SELECT_WEAPON7

    #    img, meas, rwrd, term = simulator.step(np.squeeze(ag.random_actions(1)).tolist())


    diff_y = 0
    diff_x = 0
    diff_z = 0
    inertia = 0.5
    iter = 1
    epoch = 200
    radialFlow = 10000
    radialFlowInertia = 0.01
    radialGain = 1000
    errorThresh = 1000
    updatePtsFreq = 50

    userdoc = os.path.join(os.path.expanduser("~"), "Documents")

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=500, qualityLevel=0.03, minDistance=7, blockSize=7)
    imgCentre = np.array([simulator_args['resolution'][0] / 2, simulator_args['resolution'][1] /2])
    print ("Image centre: ", imgCentre)

    while not term:
        if curr_step < agent_args['history_length']:
            curr_act = np.squeeze(ag.random_actions(1)).tolist()
            img, meas, rwrd, term = simulator.step(curr_act)
            p0 = cv2.goodFeaturesToTrack(img, mask=None, **feature_params)

        else:
            state_imgs = np.transpose(np.reshape(img_buffer[np.arange(curr_step-agent_args['history_length'], curr_step) % agent_args['history_length']], (1,) + agent_args['state_imgs_shape']), [0,2,3,1])

            state_imgs = np.transpose\
                (np.reshape
                 (img_buffer[
                      np.arange(curr_step-agent_args['history_length'], curr_step) % agent_args['history_length']
                  ],
                  (1,) + agent_args['state_imgs_shape']), [0,2,3,1])
            state_meas = np.reshape(meas_buffer[np.arange(curr_step-agent_args['history_length'], curr_step) % agent_args['history_length']], (1,agent_args['history_length']*simulator.num_meas))
#            img1 = np.sqrt(img)
            hack1 = img_buffer[(curr_step-2) % agent_args['history_length'],0,:,:]
            hack2 = img_buffer[(curr_step-1) % agent_args['history_length'],0,:,:]
#            print ("imgs shape: ", img.shape, " meas shape: ", state_meas.shape, " hack shape: ", hack1.shape, " buf shape: ", img_buffer.shape)
#            print("hacktype: ", hack1.dtype, " imgtype: ", img.dtype)
            if(curr_step % updatePtsFreq == 0):
                p0 = cv2.goodFeaturesToTrack(hack1, mask=None, **feature_params)

            p1, st, err = cv2.calcOpticalFlowPyrLK(hack1, hack2, p0, None, **lk_params)
#            print ("flat imgs shape: ", np.ndarray.flatten(state_imgs).shape, " flat meas shape: ", np.ndarray.flatten(state_meas).shape)
#            print ("meas shape: ", state_meas.shape)
            flow = (p1 - p0)[:,0,:]
#            print (len(p0), " ", flow.shape, " ", p0.shape, " ", p1.shape)
#            radialFlow = np.transpose(flow).dot(p0[:,0,:])
#            print ("FLOW ", radialFlow)
            radialFlowTmp = 0
            for i in range(0, len(p0)):
                radialFlowTmp += ((p0[i,0,:] - imgCentre)).dot(flow[i,:])


#            for vec0, vec1 in zip(flow, p0[:,0,:]):
#                radialFlowTmp += (vec1 - imgCentre).dot(vec0)


            curr_act = np.squeeze(ag.random_actions(1)[0]).tolist()
            if curr_act[:6] in acts_to_replace:
                curr_act = replacement_act
            hack = [0] * len(curr_act)

            hack[6] = diff_x
            hack[8] = -diff_y * 0.2
            hack[3] = 0  # diff_z
            #            hack[6] = 1
            #            hack[8] = 1
            curr_act[2] = 0
            curr_act[3] =  6

            radialFlow = radialFlow + radialFlowInertia * (radialFlowTmp - radialFlow)
            expectFlow = radialGain * act_buffer[(curr_step-2) % agent_args['history_length']][3]
            flowError = act_buffer[(curr_step-2) % agent_args['history_length']][3] * (expectFlow - radialFlow)
            if (flowError > errorThresh):
                print ("** FLOW ERR **")
            else:
                print ("FLOW ", radialFlow, "num pts ", len(p0))

            img, meas, rwrd, term = simulator.step(curr_act)
            if (not (meas is None)) and meas[0] > 30.:
                meas[0] = 30.
            if (not (img is None)):

#                print ("state_imgs: ", np.shape(state_imgs), "state_meas: ", np.shape(state_meas), "curr_act: ", np.shape(curr_act))
#                print ("img type: ", np.ndarray.flatten(ag.preprocess_input_images(img)).dtype, "state_img type: ", state_imgs.dtype, "state_meas type: ", state_meas.dtype)

                ag.act_ffnet(np.ndarray.flatten(state_imgs), np.ndarray.flatten(state_meas),
                             np.array(np.ndarray.flatten(act_buffer), dtype='float64'), np.ndarray.flatten(ag.preprocess_input_images(img)))
                diff_image = np.absolute(np.reshape(np.array(ag.ext_ffnet_output),
                                                    [img.shape[0], img.shape[1]]) - ag.preprocess_input_images(img))
                diff_image = np.absolute(ag.preprocess_input_images(img_buffer[(curr_step-1) % agent_args['history_length']] - ag.preprocess_input_images(img)))
                diff_image = ag.preprocess_input_images(img)

                diff_x = diff_x + inertia * (
                (np.argmax(diff_image.sum(axis=0)) / float(diff_image.shape[1])) - 0.5 - diff_x)
                diff_y = diff_x + inertia * (
                (np.argmax(diff_image.sum(axis=1)) / float(diff_image.shape[0])) - 0.5 - diff_y)

#                print ("diff_x: ", diff_x, " diff_y: ", hack[6], "centre_x: ", np.argmax(diff_image.sum(axis=0)), "centre_y: ", np.argmax(diff_image.sum(axis=1)))

                if (curr_step % epoch == 0):
                    print("saving...")
                    np.save(os.path.join('/home/paul', "hack"),
                            np.reshape(np.array(ag.ext_ffnet_output), [img.shape[0], img.shape[1]]))
                    np.save(os.path.join('/home/paul', "target"), ag.preprocess_input_images(img))
                    np.save(os.path.join('/home/paul', "diff"), diff_image)
                    diff_x = np.random.normal(0, 2)
                    diff_z = np.random.normal(10, 5)

        if not term:
            img_buffer[curr_step % agent_args['history_length']] = img
            meas_buffer[curr_step % agent_args['history_length']] = meas
            act_buffer[curr_step % agent_args['history_length']] = curr_act[:6]
            curr_step += 1

    simulator.close_game()
    ag.save('/home/paul/Dev/GameAI/vizdoom_cig2017/icolearner/ICO1/checkpoints/' + 'hack-' + str(iter))


if __name__ == '__main__':
    main()