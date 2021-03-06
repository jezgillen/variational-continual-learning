#!/usr/bin/python3
import numpy as np
import tensorflow as tf
import gzip
import pickle
import sys
from keras.datasets import mnist
sys.path.extend(['alg/'])
import vcl
import coreset
import utils
from copy import deepcopy

class PermutedMnistGenerator():
    def __init__(self, max_iter=10):
        (X, Y), (X_test, Y_test) = mnist.load_data()

        self.X_train = np.reshape(X, (-1, 28*28))
        self.Y_train = Y
        self.X_test =  np.reshape(X_test, (-1, 28*28))
        self.Y_test = Y_test

        self.max_iter = max_iter
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            np.random.seed(self.cur_iter)
            perm_inds = list(range(self.X_train.shape[1]))
            np.random.shuffle(perm_inds)

            # Retrieve train data
            next_x_train = deepcopy(self.X_train)
            next_x_train = next_x_train[:,perm_inds]
            next_y_train = np.eye(10)[self.Y_train]

            # Retrieve test data
            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:,perm_inds]
            next_y_test = np.eye(10)[self.Y_test]

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

hidden_size = [100, 100]
batch_size = 256
no_epochs = 10 # was 100, just making it faster
single_head = True
num_tasks = 5

# Run vanilla VCL
tf.set_random_seed(12)
np.random.seed(1)

coreset_size = 0
data_gen = PermutedMnistGenerator(num_tasks)
vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
    coreset.rand_from_batch, coreset_size, batch_size, single_head)
print(vcl_result)

# Run random coreset VCL
tf.reset_default_graph()
tf.set_random_seed(12)
np.random.seed(1)

coreset_size = 200
data_gen = PermutedMnistGenerator(num_tasks)
rand_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
    coreset.rand_from_batch, coreset_size, batch_size, single_head)
print(rand_vcl_result)

# Run k-center coreset VCL
tf.reset_default_graph()
tf.set_random_seed(12)
np.random.seed(1)

data_gen = PermutedMnistGenerator(num_tasks)
kcen_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
    coreset.k_center, coreset_size, batch_size, single_head)
print(kcen_vcl_result)

# Plot average accuracy
vcl_avg = np.nanmean(vcl_result, 1)
rand_vcl_avg = np.nanmean(rand_vcl_result, 1)
kcen_vcl_avg = np.nanmean(kcen_vcl_result, 1)
utils.plot('results/permuted.jpg', vcl_avg, rand_vcl_avg, kcen_vcl_avg)
