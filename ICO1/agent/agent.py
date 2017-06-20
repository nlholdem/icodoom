from __future__ import print_function
import numpy as np
import tensorflow as tf
from icolearner import IcoLearner
from icoreflex import IcoReflex

import cv2
import os
import itertools as it

class Agent:

    def __init__(self, args):
        '''Agent - powered by neural nets, can infer, act, train, test.
        '''

        # input data properties
        self.state_imgs_shape = args['state_imgs_shape']
        self.state_meas_shape = args['state_meas_shape']
        self.meas_for_net = args['meas_for_net']
        self.meas_for_manual = args['meas_for_manual']
        self.history_length = args['history_length']
        self.n_actions = args['n_actions']
        self.num_channel = args['num_channels']

        # preprocessing

        self.preprocess_input_images = args['preprocess_input_images']
        self.preprocess_input_measurements = args['preprocess_input_measurements']

        # net parameters
        self.n_ffnet_act = args['n_ffnet_act']
        self.n_ffnet_meas = args['n_ffnet_meas']
        self.n_ffnet_hidden = args['n_ffnet_hidden']
        print ("** hidden: ", self.n_ffnet_hidden)
        self.actions = np.zeros(self.n_actions)

        #        self.n_ffnet_inputs = args['n_ffnet_inputs']
#        self.n_ffnet_outputs = args['n_ffnet_outputs']

#        self.build_model()
        self.epoch = 50
        self.iter = 1

        self.radialFlowLeft = 30.
        self.radialFlowRight = 30.
        self.radialFlowInertia = 0.4
        self.radialGain = 4.
        self.rotationGain = 50.
        self.errorThresh = 10.
        self.updatePtsFreq = 50
        self.skipImage = 1
        self.skipImageICO = 5
        self.reflexGain = 0.01
        self.oldHealth = 0.

        # create masks for left and right visual fields - note that these only cover the upper half of the image
        # this is to help prevent the tracking getting confused by the floor pattern
        self.width = args['resolution'][0]
        self.height = args['resolution'][1]
        self.maskLeft = np.zeros([self.height, self.width], np.uint8)
        self.half_height = round(self.height / 2)
        self.half_width = round(self.width / 2)
        self.maskLeft[self.half_height:, :self.half_width] = 1.
        self.maskRight = np.zeros([self.height, self.width], np.uint8)
        self.maskRight[self.half_height:, self.half_width:] = 1.

        # build the ICO controllers
#        self.icoSteer = Icolearning(num_inputs= 1 + args['resolution'][0] * args['resolution'][1] + 7, num_filters=2, learning_rate=1e-5, filter_type='IIR', freqResp='band')
#        self.icoDetect = Icolearning(num_inputs= 1 + args['resolution'][0] * args['resolution'][1] + 7, num_filters=2, learning_rate=1e-5, filter_type='IIR', freqResp='band')

        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=500, qualityLevel=0.03, minDistance=7, blockSize=7)
        self.imgCentre = np.array([args['resolution'][0] / 2, args['resolution'][1] / 2])
        print("Image centre: ", self.imgCentre)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def make_ffnet(self):
        n_ffnet_inputs = self.state_imgs_shape[0] * self.state_imgs_shape[1] * self.state_imgs_shape[
            2] + self.n_ffnet_act + self.n_ffnet_meas
        n_ffnet_outputs = self.state_imgs_shape[1] * self.state_imgs_shape[2]
        print("n_ffnet_act: ", self.n_ffnet_act)
        print("n_ffnet_meas: ", self.n_ffnet_meas)
        print("ffnet: in: ", n_ffnet_inputs)
        print("ffnet: hid: ", self.n_ffnet_hidden)
        print("ffnet: out: ", n_ffnet_outputs)

        self.ffnet_input = tf.placeholder(tf.float32, shape=[None, n_ffnet_inputs])
        self.ffnet_output = tf.placeholder(tf.float32, shape=[None, n_ffnet_outputs])
        self.ffnet_target = tf.placeholder(tf.float32, shape=[None, n_ffnet_outputs])

        W_layer1 = self.weight_variable([n_ffnet_inputs, self.n_ffnet_hidden[0]])
        b_layer1 = self.bias_variable([self.n_ffnet_hidden[0]])

        W_layer2 = self.weight_variable([self.n_ffnet_hidden[0], self.n_ffnet_hidden[1]])
        b_layer2 = self.bias_variable([self.n_ffnet_hidden[1]])

        W_layer3 = self.weight_variable([self.n_ffnet_hidden[1], n_ffnet_outputs])
        b_layer3 = self.bias_variable([n_ffnet_outputs])

        h_1 = tf.nn.relu(tf.matmul(self.ffnet_input, W_layer1) + b_layer1)
        h_2 = tf.nn.relu(tf.matmul(h_1, W_layer2) + b_layer2)

        # dropout
        self.keep_prob = tf.placeholder(tf.float32)
        my_drop = tf.nn.dropout(h_2, self.keep_prob)
        #        print("output shape: ", self.ffnet_output.get_shape(), "target shape: ", self.ffnet_target.get_shape())
        #        print("W3: ", W_layer3.get_shape(), " bias3: ", b_layer3.get_shape())

        self.ffnet_output = tf.matmul(h_2, W_layer3) + b_layer3
        #        print("output shape: ", self.ffnet_output.get_shape(), "target shape: ", self.ffnet_target.get_shape())
        #        print("W3: ", W_layer3.get_shape(), " bias3: ", b_layer3.get_shape())

        self.loss = tf.squared_difference(self.ffnet_output, self.ffnet_target)

        self.ffnet_train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        self.accuracy = tf.reduce_mean(self.loss)
        #        sess.run(tf.global_variables_initializer())

    def build_model(self):

        # make the actual net
        self.make_ffnet()
        tf.initialize_all_variables().run(session=self.sess)

        # make the saver, lr and param summaries
#        self.saver = tf.train.Saver()


    def act(self, state_imgs, state_meas, curr_step):

        if curr_step < agent_args['history_length']:
            self.p0Left = cv2.goodFeaturesToTrack(img, mask=maskLeft, **feature_params)
            self.p0Right = cv2.goodFeaturesToTrack(img, mask=maskRight, **feature_params)


        else:

            if (curr_step % updatePtsFreq == 0):
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
            expectFlowLeft = radialGain * forward + (rotationGain * rotation if rotation < 0. else 0.)
            expectFlowRight = radialGain * forward - (rotationGain * rotation if rotation > 0. else 0.)
            flowErrorLeft = forward * (expectFlowLeft - radialFlowLeft) / (1. + rotationGain * np.abs(rotation))
            flowErrorRight = forward * (expectFlowRight - radialFlowRight) / (1. + rotationGain * np.abs(rotation))

            if curr_step > 100:

                health = meas[1]
                healthChange = health-oldHealth if health<oldHealth else 0.

                icoInSteer = self.reflexGain * ((flowErrorRight - self.errorThresh) if (flowErrorRight - self.errorThresh) > 0. else 0. - (flowErrorLeft - self.errorThresh) if (flowErrorLeft - self.errorThresh) > 0. else 0. )

                diff_theta = .0 * max(min(icoControlSteer, 5.), -5.)

                curr_act = np.zeros(7).tolist()

                curr_act[0] = 0
                curr_act[1] = 0
                curr_act[2] = 0
                curr_act[3] = curr_act[3] + diff_z
                curr_act[3] = 30.
                curr_act[4] = 0
                curr_act[5] = 0

                curr_act[6] = curr_act[6] + diff_theta
                #            curr_act[6] = 0.
                self.oldHealth = health

        return curr_act

    def act_ffnet(self, in_image, in_meas, in_actions, target_image):
#        print ("ACT: img: ", in_image.shape)
#        print ("ACT: meas: ", in_meas.shape)
#        print ("ACT: act: ", in_actions.shape)
#        print ("ACT: targ: ", target_image.shape)

        in_length = in_image.shape[0] + in_meas.shape[0] + in_actions.shape[0]
        net_inputs = np.reshape(np.concatenate([in_image, in_meas, in_actions], axis=0), (1, in_length))
        net_targets = np.reshape(target_image, (1, target_image.shape[0]))
#        print ("Health: ", in_meas[1])

        if (in_meas[1] > 1.0): # don't train on images where the player is dead
            with self.sess.as_default():
                self.ext_ffnet_output, hack = self.sess.run([self.ffnet_output, self.ffnet_train_step], feed_dict={
                    self.ffnet_input: net_inputs,
                    self.ffnet_target: net_targets, self.keep_prob: 0.5
                })

                if self.iter % self.epoch == 0:
                    print ("LOSS: ", self.accuracy.eval(feed_dict={
                        self.ffnet_input: net_inputs,
                        self.ffnet_target: net_targets, self.keep_prob: 0.5
                    }))

            self.iter = self.iter+1

    def act_net(self, state_imgs, state_meas, objective):
        #Act given a state and objective
        predictions = self.sess.run(self.pred_all, feed_dict={self.input_images: state_imgs,
                                                            self.input_measurements: state_meas[:,self.meas_for_net]})
        #print (predictions)

        objectives = np.sum(predictions[:,:,objective[0]]*objective[1][None,None,:], axis=2)
        curr_action = np.argmax(objectives, axis=1)
#        print (" ** ACTION ", curr_action)
        return curr_action

    # act_manual is a purely hard-coded method to handle weapons-switching
    def act_manual(self, state_meas):
        if len(self.meas_for_manual) == 0:
            return []
        else:
            assert(len(self.meas_for_manual) == 13) # expected to be [AMMO2 AMMO3 AMMO4 AMMO5 AMMO6 AMMO7 WEAPON2 WEAPON3 WEAPON4 WEAPON5 WEAPON6 WEAPON7 SELECTED_WEAPON]
            assert(self.num_manual_controls == 6) # expected to be [SELECT_WEAPON2 SELECT_WEAPON3 SELECT_WEAPON4 SELECT_WEAPON5 SELECT_WEAPON6 SELECT_WEAPON7]

            curr_act = np.zeros((state_meas.shape[0],self.num_manual_controls), dtype=np.int)
            for ns in range(state_meas.shape[0]):
                # always pistol
                #if not state_meas[ns,self.meas_for_manual[12]] == 2:
                    #curr_act[ns, 0] = 1
                # best weapon
                curr_ammo = state_meas[ns,self.meas_for_manual[:6]]
                curr_weapons = state_meas[ns,self.meas_for_manual[6:12]]
                #print(curr_ammo,curr_weapons)
                available_weapons = np.logical_and(curr_ammo >= np.array([1,2,1,1,1,40]), curr_weapons)
                if any(available_weapons):
                    best_weapon = np.nonzero(available_weapons)[0][-1]
                    if not state_meas[ns,self.meas_for_manual[12]] == best_weapon+2:
                        curr_act[ns, best_weapon] = 1
            return curr_act

    def save(self, checkpoint_dir):
        with self.sess.as_default():
            save_path = self.saver.save(self.sess, checkpoint_dir)
            print ("saving model file: ", save_path)



    def load(self, checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


