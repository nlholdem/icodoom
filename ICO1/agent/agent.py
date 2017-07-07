from __future__ import print_function
import numpy as np
import time
import tensorflow as tf
import ops as my_ops
import os
import re
import itertools as it


class Agent:
    def __init__(self, sess, args):
        '''Agent - powered by neural nets, can infer, act, train, test.
        '''
        self.sess = sess

        # input data properties
        self.state_imgs_shape = args['state_imgs_shape']
        self.state_meas_shape = args['state_meas_shape']
        self.meas_for_net = args['meas_for_net']
        self.meas_for_manual = args['meas_for_manual']

        # preprocessing
        self.preprocess_input_images = args['preprocess_input_images']
        self.preprocess_input_measurements = args['preprocess_input_measurements']

        # net parameters
        self.conv_params = args['conv_params']
        self.fc_img_params = args['fc_img_params']
        self.fc_meas_params = args['fc_meas_params']
        self.fc_joint_params = args['fc_joint_params']
        self.target_dim = args['target_dim']

        self.n_ffnet_inputs = args['n_ffnet_inputs']
        self.n_ffnet_hidden = args['n_ffnet_hidden']
        self.n_ffnet_outputs = args['n_ffnet_outputs']
        self.n_ffnet_lr = args['learning_rate']

        self.build_model()
        self.epoch = 500
        self.iter = 1

    def prepare_controls_and_actions(self):
        self.discrete_controls_to_net = np.array(
            [i for i in range(len(self.discrete_controls)) if not i in self.discrete_controls_manual])
        self.num_manual_controls = len(self.discrete_controls_manual)

        self.net_discrete_actions = []
        if not self.opposite_button_pairs:
            for perm in it.product([False, True], repeat=len(self.discrete_controls_to_net)):
                self.net_discrete_actions.append(list(perm))
        else:
            for perm in it.product([False, True], repeat=len(self.discrete_controls_to_net)):
                # remove actions where both opposite buttons are pressed
                act = list(perm)
                valid = True
                for bp in self.opposite_button_pairs:
                    if act[bp[0]] == act[bp[1]] == True:
                        valid = False
                if valid:
                    self.net_discrete_actions.append(act)

        self.num_net_discrete_actions = len(self.net_discrete_actions)
        print("num_net_discrete_actions: ", self.num_net_discrete_actions)
        print("net_discrete_actions: ", self.net_discrete_actions)
        self.action_to_index = {tuple(val): ind for (ind, val) in enumerate(self.net_discrete_actions)}
        self.net_discrete_actions = np.array(self.net_discrete_actions)
        self.onehot_discrete_actions = np.eye(self.num_net_discrete_actions)

    def preprocess_actions(self, acts):
        to_net_acts = acts[:, self.discrete_controls_to_net]
        return self.onehot_discrete_actions[
            np.array([self.action_to_index[tuple(act)] for act in to_net_acts.tolist()])]

    def postprocess_actions(self, acts_net, acts_manual=[]):
        out_actions = np.zeros((acts_net.shape[0], len(self.discrete_controls)), dtype=np.int)

        out_actions[:, self.discrete_controls_to_net] = self.net_discrete_actions[acts_net]
        # print(acts_net, acts_manual, self.discrete_controls_to_net, out_actions)
        if len(acts_manual):
            out_actions[:, self.discrete_controls_manual] = acts_manual
        # print ("out_actions: ", out_actions)
        return out_actions

    def random_actions(self, num_samples):
        acts_net = np.random.randint(0, self.num_net_discrete_actions, num_samples)
        acts_manual = np.zeros((num_samples, self.num_manual_controls),
                               dtype=np.bool)  # don't switch weapons in random mode
        return self.postprocess_actions(acts_net, acts_manual)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=1e-4)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def make_ffnet(self):

        n_ffnet_inputs = self.n_ffnet_inputs
        n_ffnet_outputs = self.n_ffnet_outputs
        n_ffnet_outputs = 1

        self.ffnet_input = tf.placeholder(tf.float32, shape=[1, n_ffnet_inputs])
        ffnet_output = tf.placeholder(tf.float32, shape=[1, n_ffnet_outputs])
        self.ffnet_target = tf.placeholder(tf.float32, shape=[1, n_ffnet_outputs])

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
        print("input shape: ", self.ffnet_input.get_shape(), "output shape: ", ffnet_output.get_shape(), "target shape: ", self.ffnet_target.get_shape())
        print("W3: ", W_layer3.get_shape(), " bias3: ", b_layer3.get_shape())

        ffnet_output = tf.matmul(h_2, W_layer3) + b_layer3
        print("output shape: ", ffnet_output.get_shape(), "target shape: ", self.ffnet_target.get_shape())
        print("W3: ", W_layer3.get_shape(), " bias3: ", b_layer3.get_shape())
        print("learning rate: ", self.n_ffnet_lr)

        self.net_output = ffnet_output
        self.loss = tf.squared_difference(ffnet_output, self.ffnet_target)

        self.ffnet_train_step = tf.train.AdamOptimizer(self.n_ffnet_lr).minimize(self.loss)

        self.accuracy = tf.reduce_mean(self.loss)

    #        sess.run(tf.global_variables_initializer())


    def make_net(self, input_images, input_measurements, input_actions, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        self.fc_val_params = np.copy(self.fc_joint_params)
        self.fc_val_params['out_dims'][-1] = self.target_dim
        self.fc_adv_params = np.copy(self.fc_joint_params)
        self.fc_adv_params['out_dims'][-1] = len(self.net_discrete_actions) * self.target_dim
        print(len(self.net_discrete_actions) * self.target_dim)
        p_img_conv = my_ops.conv_encoder(input_images, self.conv_params, 'p_img_conv', msra_coeff=0.9)
        print("Conv Params: ", self.conv_params)

        p_img_fc = my_ops.fc_net(my_ops.flatten(p_img_conv), self.fc_img_params, 'p_img_fc', msra_coeff=0.9)
        print("img_params", self.fc_img_params)
        p_meas_fc = my_ops.fc_net(input_measurements, self.fc_meas_params, 'p_meas_fc', msra_coeff=0.9)
        print("meas_params", self.fc_meas_params)
        p_val_fc = my_ops.fc_net(tf.concat(1, [p_img_fc, p_meas_fc]), self.fc_val_params, 'p_val_fc', last_linear=True,
                                 msra_coeff=0.9)
        print("val_params", self.fc_val_params)
        p_adv_fc = my_ops.fc_net(tf.concat(1, [p_img_fc, p_meas_fc]), self.fc_adv_params, 'p_adv_fc', last_linear=True,
                                 msra_coeff=0.9)
        print("adv_params", self.fc_adv_params)

        p_adv_fc_nomean = p_adv_fc - tf.reduce_mean(p_adv_fc, reduction_indices=1, keep_dims=True)

        self.pred_all_nomean = tf.reshape(p_adv_fc_nomean, [-1, len(self.net_discrete_actions), self.target_dim])
        self.pred_all = self.pred_all_nomean + tf.reshape(p_val_fc, [-1, 1, self.target_dim])
        self.pred_relevant = tf.boolean_mask(self.pred_all, tf.cast(input_actions, tf.bool))
        print("make_net: input_actions: ", input_actions)
        print("make_net: pred_all: ", self.pred_all)
        print("make_net: pred_relevant: ", self.pred_relevant)

    def build_model(self):
        # prepare the data
        print("state images shape: ", self.state_imgs_shape[0], self.state_imgs_shape[1], self.state_imgs_shape[2])
        self.input_images = tf.placeholder(tf.float32, [None] + [self.state_imgs_shape[1], self.state_imgs_shape[2],
                                                                 self.state_imgs_shape[0]],
                                           name='input_images')
#        self.input_measurements = tf.placeholder(tf.float32, [None] + list(self.state_meas_shape),
#                                                 name='input_measurements')
#        self.input_actions = tf.placeholder(tf.float32, [None, self.num_net_discrete_actions],
#                                            name='input_actions')

        if self.preprocess_input_images:
            self.input_images_preprocessed = self.preprocess_input_images(self.input_images)
#        if self.preprocess_input_measurements:
#            self.input_measurements_preprocessed = self.preprocess_input_measurements(self.input_measurements)

            # make the actual net
        #        self.make_net(self.input_images_preprocessed, self.input_measurements_preprocessed, self.input_actions)
        self.make_ffnet()

        # make the saver, lr and param summaries
        self.saver = tf.train.Saver()

        tf.initialize_all_variables().run(session=self.sess)

    def act(self, state_imgs, state_meas, objective):
        return self.postprocess_actions(self.act_net(state_imgs, state_meas, objective), self.act_manual(
            state_meas)), None  # last output should be predictions, but we omit these for now

    def learn_ffnet(self, inputs, target):

        net_inputs = inputs
        net_targets = target

        self.ffnet_train_step.run(session=self.sess, feed_dict={
            self.ffnet_input: net_inputs,
            self.ffnet_target: net_targets, self.keep_prob: 0.5
        })
        net_output = self.net_output.eval(session=self.sess, feed_dict={
                self.ffnet_input: net_inputs,
                self.ffnet_target: net_targets, self.keep_prob: 0.5
            })
        if self.iter % self.epoch == 0:
            print("LOSS: ", self.accuracy.eval(session=self.sess, feed_dict={
                self.ffnet_input: net_inputs,
                self.ffnet_target: net_targets, self.keep_prob: 0.5
            }))

        self.iter = self.iter + 1
        return net_output

    def test_ffnet(self, inputs):

        net_inputs = inputs

        net_output = self.net_output.eval(session=self.sess, feed_dict={
                self.ffnet_input: net_inputs
            })

        self.iter = self.iter + 1
        return net_output

    def act_net(self, state_imgs, state_meas, objective):
        # Act given a state and objective
        predictions = self.sess.run(self.pred_all, feed_dict={self.input_images: state_imgs,
                                                              self.input_measurements: state_meas[:,
                                                                                       self.meas_for_net]})
        # print (predictions)

        objectives = np.sum(predictions[:, :, objective[0]] * objective[1][None, None, :], axis=2)
        curr_action = np.argmax(objectives, axis=1)
        #        print (" ** ACTION ", curr_action)
        return curr_action

    # act_manual is a purely hard-coded method to handle weapons-switching
    def act_manual(self, state_meas):
        if len(self.meas_for_manual) == 0:
            return []
        else:
            assert (len(
                self.meas_for_manual) == 13)  # expected to be [AMMO2 AMMO3 AMMO4 AMMO5 AMMO6 AMMO7 WEAPON2 WEAPON3 WEAPON4 WEAPON5 WEAPON6 WEAPON7 SELECTED_WEAPON]
            assert (
            self.num_manual_controls == 6)  # expected to be [SELECT_WEAPON2 SELECT_WEAPON3 SELECT_WEAPON4 SELECT_WEAPON5 SELECT_WEAPON6 SELECT_WEAPON7]

            curr_act = np.zeros((state_meas.shape[0], self.num_manual_controls), dtype=np.int)
            for ns in range(state_meas.shape[0]):
                # always pistol
                # if not state_meas[ns,self.meas_for_manual[12]] == 2:
                # curr_act[ns, 0] = 1
                # best weapon
                curr_ammo = state_meas[ns, self.meas_for_manual[:6]]
                curr_weapons = state_meas[ns, self.meas_for_manual[6:12]]
                # print(curr_ammo,curr_weapons)
                available_weapons = np.logical_and(curr_ammo >= np.array([1, 2, 1, 1, 1, 40]), curr_weapons)
                if any(available_weapons):
                    best_weapon = np.nonzero(available_weapons)[0][-1]
                    if not state_meas[ns, self.meas_for_manual[12]] == best_weapon + 2:
                        curr_act[ns, best_weapon] = 1
            return curr_act

    def load(self, checkpoint_dir):
        print("ckpt_name", checkpoint_dir)

        with self.sess.as_default():
            self.saver.restore(self.sess, checkpoint_dir)

#        saver = tf.train.import_meta_graph(checkpoint_dir)
#        saver.restore(self.sess, tf.train.latest_checkpoint('./'))



    def save(self, _dir, iter):
        with self.sess.as_default():
            save_path = self.saver.save(self.sess, _dir, global_step=iter)
            print("saving model file: ", save_path)

