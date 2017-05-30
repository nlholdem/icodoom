from __future__ import print_function
import numpy as np
import time
import tensorflow as tf
import ops as my_ops
import os
import re
import itertools as it


class Nnet:
    def __init__(self, sess, args):

        self.sess = sess
        self.n_hidden = args['n_hidden']
        self.imgs_shape = args['imgs_shape']
        self.output = np.zeros(args['imgs_shape'][0] * args['imgs_shape'][1])
        self.actions = np.zeros(args['n_actions'])
        self.game_vars = np.zeros(args['n_game_vars'])
        self.n_inputs = self.imgs_shape[0] * self.imgs_shape[1] * self.imgs_shape[2] + args['n_actions'] + args['n_game_vars']
        self.n_outputs = self.imgs_shape[1] * self.imgs_shape[2]
        print ("Imgs shape: ", self.imgs_shape)
        print ("Num inputs: ", self.n_inputs)
        print ("Num hidden: ", self.n_hidden[0], self.n_hidden[1])
        print ("Num outputs: ", self.n_outputs)

        self.epoch = 50
        self.iter = 1

    def make_ffnet(self):

        self.ffnet_input = tf.placeholder(tf.float32, shape=[None, self.n_inputs])
        self.ffnet_output = tf.placeholder(tf.float32, shape=[None, self.n_outputs])
        self.ffnet_target = tf.placeholder(tf.float32, shape=[None, self.n_outputs])

        W_layer1 = self.weight_variable([self.n_inputs, self.n_hidden[0]])
        b_layer1 = self.bias_variable([self.n_hidden[0]])

        W_layer2 = self.weight_variable([self.n_hidden[0], self.n_hidden[1]])
        b_layer2 = self.bias_variable([self.n_hidden[1]])

        W_layer3 = self.weight_variable([self.n_hidden[1], self.n_outputs])
        b_layer3 = self.bias_variable([self.n_outputs])

        h_1 = tf.nn.relu(tf.matmul(self.ffnet_input, W_layer1) + b_layer1)
        h_2 = tf.nn.relu(tf.matmul(h_1, W_layer2) + b_layer2)

        # dropout
        self.keep_prob = tf.placeholder(tf.float32)
        my_drop = tf.nn.dropout(h_2, self.keep_prob)
        print("output shape: ", self.ffnet_output.get_shape(), "target shape: ", self.ffnet_target.get_shape())
        print("W3: ", W_layer3.get_shape(), " bias3: ", b_layer3.get_shape())

        self.ffnet_output = tf.matmul(h_2, W_layer3) + b_layer3
        print("output shape: ", self.ffnet_output.get_shape(), "target shape: ", self.ffnet_target.get_shape())
        print("W3: ", W_layer3.get_shape(), " bias3: ", b_layer3.get_shape())

        self.loss = tf.squared_difference(self.ffnet_output, self.ffnet_target)

        self.ffnet_train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        self.accuracy = tf.reduce_mean(self.loss)
        #        sess.run(tf.global_variables_initializer())


    def act_ffnet(self, in_image, in_meas, in_actions, target_image):

        in_length = in_image.shape[0] + in_meas.shape[0] + in_actions.shape[0]
        net_inputs = np.reshape(np.concatenate([in_image, in_meas, in_actions], axis=0), (1, in_length))
        net_targets = np.reshape(target_image, (1, target_image.shape[0]))
        #        print ("Health: ", in_meas[1])

        if (in_meas[1] > 1.0):  # don't train on images where the player is dead
            with self.sess.as_default():
                self.ext_ffnet_output, hack = self.sess.run([self.ffnet_output, self.ffnet_train_step], feed_dict={
                    self.ffnet_input: net_inputs,
                    self.ffnet_target: net_targets, self.keep_prob: 0.5
                })

                if self.iter % self.epoch == 0:
                    print("LOSS: ", self.accuracy.eval(feed_dict={
                        self.ffnet_input: net_inputs,
                        self.ffnet_target: net_targets, self.keep_prob: 0.5
                    }))

            self.iter = self.iter + 1

    def weight_variable(self, shape):
        print ("weight_variable: shape: ", shape)
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)




