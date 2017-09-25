from __future__ import print_function
import numpy as np
import sys
sys.path.append('./bin/python')
import os.path
import vizdoom
from agent.doom_simulator import DoomSimulator
from agent.agent import Agent
from agent.nnet import visPredictor
import tensorflow as tf
import cv2


def buildFilters():
    ksize = 35
    sigma = 5.
    gamma = 1.
    theta_vals = np.linspace(0., np.pi, 4, endpoint=False)
#    lambd_vals = (3, 7, 13, 27)
#    sigma_vals = (1, 3, 7, 15)
    lambd_vals = (1.5, 3)
    sigma_vals = (0.5, 1.5)

    """
    theta: orientation
    lambda: wavelength
    sigma: standard deviation
    gamma: aspect ratio
    """
    coeffs = ((theta, lambd, sigma) for lambd, sigma in zip(lambd_vals, sigma_vals) for theta in theta_vals)

    filters = [(cv2.getGaborKernel((ksize,ksize), coeff[2], coeff[0], coeff[1], gamma)/(0.01*ksize*ksize*sigma), coeff[2])
               for coeff in coeffs]

    for f in filters:
        print("Spatial filter: Min: ", np.amin(f[0]), " Max: ", np.amax(f[0]))

    return filters

def getMaxColourPos(img, colour):
    img = np.array(img, dtype='float64')
    #print ("getMaxColourPos: image shape: ", img.shape)
    width = int(img.shape[1])
    height = int(img.shape[0])
#    img[:,10,10] = [0,0,255]
    diff = np.ones(img.shape)
    diff[:,:,0] = colour[0]
    diff[:,:,1] = colour[1]
    diff[:,:,2] = colour[2]
    diff = np.absolute(np.add(diff, (-1*img)))
    diff = np.sum(diff, axis=2)

    indx = np.argmin(diff)
    indx_y = int(indx / width)
    indx_x = indx % width

    bestColour = diff[indx_y, indx_x]
    pts = np.asarray(np.where((diff - bestColour) < 75))
    if (pts.shape[1]>0):
        bottomLeft = np.array([np.amin(pts[1]), np.amin(pts[0])])
        topRight = np.array([np.amax(pts[1]), np.amax(pts[0])])
    else:
        bottomLeft = []
        topRight = []
#    print("COLOUR: ", [indx_x, indx_y])
    return np.array([indx_x, indx_y]), bottomLeft, topRight, np.mean(diff) - diff[indx_y,indx_x]



def main():
    
    ## Simulator
    simulator_args = {}
    simulator_args['config'] = 'config/config.cfg'
    simulator_args['resolution'] = (320,240)
    simulator_args['frame_skip'] = 1
    simulator_args['color_mode'] = 'RGB24'
    simulator_args['game_args'] = "+name IntelAct +colorset 7"
    width = simulator_args['resolution'][0]
    height = simulator_args['resolution'][1]
    imgCentre = np.array([int(width / 2), int(height /2)])

    ## Agent    
    agent_args = {}
    
    # preprocessing
    agent_args['preprocess_input_images'] = lambda x: x / 255. - 0.5
    agent_args['preprocess_input_measurements'] = lambda x: x / 100. - 0.5
    agent_args['num_future_steps'] = 6
    pred_scale_coeffs = np.expand_dims((np.expand_dims(np.array([8.,40.,1.]),1) * np.ones((1,agent_args['num_future_steps']))).flatten(),0)
    agent_args['postprocess_predictions'] = lambda x: x * pred_scale_coeffs
    agent_args['discrete_controls_manual'] = range(6,12) 
    agent_args['meas_for_net_init'] = range(3)
    agent_args['meas_for_manual_init'] = range(3,16)
    agent_args['opposite_button_pairs'] = [(0,1),(2,3)]
    
    # net parameters
    agent_args['conv_params']     = np.array([(16,5,4), (32,3,2), (64,3,2), (128,3,2)],
                                     dtype = [('out_channels',int), ('kernel',int), ('stride',int)])
    agent_args['fc_img_params']   = np.array([(128,)], dtype = [('out_dims',int)])
    agent_args['fc_meas_params']  = np.array([(128,), (128,), (128,)], dtype = [('out_dims',int)]) 
    agent_args['fc_joint_params'] = np.array([(256,), (256,), (-1,)], dtype = [('out_dims',int)])   
    agent_args['target_dim'] = agent_args['num_future_steps'] * len(agent_args['meas_for_net_init'])

    # simple NNet controller training with feedback-error learning
    agent_args['n_ffnet_input'] = width*height
    agent_args['n_ffnet_hidden'] = np.array([50,5])
    agent_args['n_ffnet_output'] = 1

    # experiment arguments
    agent_args['test_objective_params'] = (np.array([5,11,17]), np.array([1.,1.,1.]))
    agent_args['history_length'] = 1
    agent_args['test_checkpoint'] = 'model'
    
    print('starting simulator')

    simulator = DoomSimulator(simulator_args)
    
    print('started simulator')

    agent_args['discrete_controls'] = simulator.discrete_controls
    agent_args['continuous_controls'] = simulator.continuous_controls
    agent_args['state_imgs_shape'] = (agent_args['history_length']*simulator.num_channels, simulator.resolution[1], simulator.resolution[0])
    if 'meas_for_net_init' in agent_args:
        agent_args['meas_for_net'] = []
        for ns in range(agent_args['history_length']):
            agent_args['meas_for_net'] += [i + simulator.num_meas * ns for i in agent_args['meas_for_net_init']]
        agent_args['meas_for_net'] = np.array(agent_args['meas_for_net'])
    else:
        agent_args['meas_for_net'] = np.arange(agent_args['history_length']*simulator.num_meas)
    if len(agent_args['meas_for_manual_init']) > 0:
        agent_args['meas_for_manual'] = np.array([i + simulator.num_meas*(agent_args['history_length']-1) for i in agent_args['meas_for_manual_init']]) # current timestep is the last in the stack
    else:
        agent_args['meas_for_manual'] = []
    agent_args['state_meas_shape'] = (len(agent_args['meas_for_net']),)
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False))
    ag = Agent(sess, agent_args)

    model_path = "/home/paul/Dev/GameAI/vizdoom_cig2017/icodoom/ICO1/checkpoints/"
    if(ag.load(model_path)):
        print ("Load success")
    else:
        print ("Failed to load model")

    spatialFilters = buildFilters()
    img_buffer = np.zeros((agent_args['history_length'], simulator.num_channels, simulator.resolution[1], simulator.resolution[0]))
    meas_buffer = np.zeros((agent_args['history_length'], simulator.num_meas))
    act_buffer = np.zeros((agent_args['history_length'], 6))
    curr_step = 0
    term = False
    
    acts_to_replace = [a+b+d+e for a in [[0,0],[1,1]] for b in [[0,0],[1,1]] for d in [[0]] for e in [[0],[1]]]
    print(acts_to_replace)
    replacement_act = [0,1,0,0,0,1,0,0,0,0,0,0]
    #MOVE_FORWARD   MOVE_BACKWARD   TURN_LEFT   TURN_RIGHT  ATTACK  SPEED   SELECT_WEAPON2  SELECT_WEAPON3  SELECT_WEAPON4  SELECT_WEAPON5  SELECT_WEAPON6  SELECT_WEAPON7

#    img, meas, rwrd, term = simulator.step(np.squeeze(ag.random_actions(1)).tolist())


    diff_y = 0
    diff_x = 0
    diff_z = 0
    inertia = 0.5
    iter = 1
    epoch = 100

    gray = np.zeros((height, width))
    posImage = np.zeros((height, width))
    negImage = np.zeros((height, width))
    thresh = 0.
    img_meas = np.zeros((1,width*height))
    state_meas = np.zeros((1,simulator.num_meas))
    target_meas = np.zeros((1,1))

    userdoc = os.path.join(os.path.expanduser("~"), "Documents")
    reflex = 0.
    netOut = 0.

    while not term:
        if curr_step < agent_args['history_length']:
            curr_act = np.squeeze(ag.random_actions(1)).tolist()
            img, meas, rwrd, term = simulator.step(curr_act)

        else:

            curr_act = np.squeeze(ag.random_actions(1)[0]).tolist()
            if curr_act[:6] in acts_to_replace:
                curr_act = replacement_act

            curr_act = [0 for i in range(len(curr_act))]
            #print ("Reflex: ", reflex)
            curr_act[6] = 10.*reflex
            #print ("NetOut: ", netOut)
            curr_act[6] += 40. * netOut


            img, meas, rwrd, term = simulator.step(curr_act)
            if (not (meas is None)) and meas[0] > 30.:
                meas[0] = 30.
            if (not (img is None)):

                centre, bottomLeft, topRight, colourStrength = getMaxColourPos(img, [255, 0, 0])
                colourSteer = imgCentre[0]
                imgRect = img
                if (len(bottomLeft) > 0 and len(topRight) > 0 and ((topRight[0] - bottomLeft[0]) < width / 3) and ((topRight[1] - bottomLeft[1]) < height / 2)):
                    imgRect = cv2.rectangle(imgRect, (bottomLeft[0], bottomLeft[1]), (topRight[0], topRight[1]), (255,255,255), 2)
                    imgRect = cv2.circle(imgRect, (bottomLeft[0] + int(0.5*(topRight[0] - bottomLeft[0])),
                                                   bottomLeft[1] + int(0.5*(topRight[1] - bottomLeft[1]))),
                                         3, (255,0,0), 3)
                    colourSteer = bottomLeft[0] + int(0.5 * (topRight[0] - bottomLeft[0]))

                cv2.arrowedLine(imgRect, (colourSteer, imgCentre[1]+10), (colourSteer, imgCentre[1]), color=(0,55,255), thickness=2)
                cv2.imwrite("/home/paul/tmp/Images/Positive/rect-" + str(curr_step) + ".jpg", imgRect)

                reflex = (float(colourSteer) - float(imgCentre[0]))/float(width)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                negImage[...] = 0.
                posImage[...] = 0.
#                for f in spatialFilters:
#                    gray1 = cv2.filter2D(gray, -1, f[0])
#                    negImage[np.where(gray1 < thresh)] += gray1[np.where(gray1 < thresh)] / float(len(spatialFilters))
#                    posImage[np.where(gray1 > thresh)] += gray1[np.where(gray1 > thresh)] / float(len(spatialFilters))

                #print ("state_imgs: ", np.shape(state_imgs), "state_meas: ", np.shape(state_meas), "curr_act: ", np.shape(curr_act))
                #print ("img type: ", np.ndarray.flatten(ag.preprocess_input_images(img)).dtype, "state_img type: ", state_imgs.dtype, "state_meas type: ", state_meas.dtype)

                img_meas[0,:] = np.ndarray.flatten(gray)
                state_meas[0,:] = np.ndarray.flatten(meas)
                target_meas[:,:] = reflex - netOut
                #target_meas[:,:] = 0.
                ag.act_ffnet(img_meas, meas, target_meas)

                netOut = np.ndarray.flatten(ag.ext_ffnet_output)[0]
                #print (" netout: ", ag.ext_ffnet_output, " NetOut: ", netOut)
                print ("Step: ", curr_step, " OUT: ", ag.ext_ffnet_output, " reflex: ", reflex)


                if (iter % epoch == 0):
                    print ("saving...")
                    #ag.save('/home/paul/Dev/GameAI/vizdoom_cig2017/icodoom/ICO1/checkpoints/' + 'BP-' + str(iter))
                    ag.save(model_path+"BP-", iter)
                iter += 1

        if not term:
            curr_step += 1

    simulator.close_game()
    ag.save('/home/paul/Dev/GameAI/vizdoom_cig2017/icolearner/ICO1/checkpoints/' + 'hack-' + str(iter))


if __name__ == '__main__':
    main()
