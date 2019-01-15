#!/usr/bin/python3

import numpy as np
import tensorflow as tf
from keras.datasets import mnist

tf.enable_eager_execution()

class Neural_Net_Base():
    """
    A neural net base class, containing functions for training
    """
    def __init__(self):
        self.opt = None
        self.weights = {}

    def train(self, X, Y, num_epochs=10, batch_size=128):
        """
        Does batch SGD
        Yields control back every epoch, returning the average ELBO across batches
        """
        if(self.opt is None):
            self.opt = tf.train.AdamOptimizer(0.001)
        for i in range(num_epochs):
            loss_history = []
            for batch in range(X.shape[0]//batch_size):
                start = batch*batch_size
                end = (batch+1)*batch_size

                with tf.GradientTape() as tape:
                    loss = self.loss(X[start:end],Y[start:end])

                loss_history.append(loss.numpy())

                weights = list(self.weights.values())
                gradients = tape.gradient(loss, weights)
                self.opt.apply_gradients(zip(gradients, weights))

            yield np.mean(loss_history)

    def loss(X, Y):
        raise NotImplementedError()
    def predict(X):
        raise NotImplementedError()


class Gaussian_BBB_NN(Neural_Net_Base):
    """
    A bayesian neural network running the Bayes by Backprop algorithm for training,
    and using a normal prior and posterior.
    
    TODO - If I want accurate Variational Free Energy estimates for something other than
           minimizing, such as comparing multiple models, then I'll need to add in constants
           to the logLiklihood and KLdivergence calculations
         - Try bimodal prior and posterior
    """
    def __init__(self, num_training_samples=7, num_pred_samples=99,
                optimizer=None):

       super().__init__()
       self.num_training_samples = num_training_samples
       self.num_pred_samples = num_pred_samples
       self.opt = optimizer 
       self.default_prior_mean = 0
       self.default_prior_std = 1
       self.weight_samples = {}
       self.prior = {}

    def predict(self, X):
        logits = self.model(X, samples=self.num_pred_samples)
        predictions = tf.nn.softmax(logits)
        return tf.reduce_mean(predictions, axis=0).numpy()

    def model(self, X, samples=1):
        """
        Returns logits of the model, of shape [sample, batch, logit]
        """

        # Transforms inputs from [batch, input] into [sample, batch, input]
        X = tf.tile(tf.expand_dims(X, 0), [samples, 1, 1])        

        z = self.fully_connected_layer(X, samples, size=(28*28, 128), name="layer_one")
        h = tf.nn.relu(z)
        z = self.fully_connected_layer(h, samples, size=(128, 64), name="layer_two")
        h = tf.nn.relu(z)
        z = self.fully_connected_layer(h, samples, size=(64, 10), name="layer_three")
        logits = z
        # logits should still be of shape [sample, batch, logit]
        return logits

    def fully_connected_layer(self, x, samples, size, name):
        try:
            weights_mean = self.weights[name+'_weights_mean']
            weights_std = self.weights[name+'_weights_std']
            bias_mean = self.weights[name+'_bias_mean']
            bias_std = self.weights[name+'_bias_std']
        except KeyError:
            weights_mean = self.weights[name+'_weights_mean'] = tf.Variable(tf.zeros(size))
            weights_std = self.weights[name+'_weights_std'] = tf.Variable(tf.ones(size)/10)
            bias_mean = self.weights[name+'_bias_mean'] = tf.Variable(tf.zeros(size[-1]))
            bias_std = self.weights[name+'_bias_std'] = tf.Variable(tf.ones(size[-1])/10)

        weights_sample = tf.random_normal((samples, *size), 
                                          mean=weights_mean,
                                          stddev=weights_std)
        self.weight_samples[name+'_weights'] = weights_sample

        #TODO some transformation on the bias sample
        bias_sample = tf.random_normal((samples, x.shape[-2], size[-1]), 
                                          mean=bias_mean,
                                          stddev=bias_std)
        self.weight_samples[name+'_bias'] = bias_sample
        return x@weights_sample + bias_sample

    def loss(self, X, Y, samples=None):
        if samples is None:
            samples = self.num_training_samples
        return -self.logLikelihood(X, Y, samples) + self.KL_Divergence() 

    def logLikelihood(self, X, Y, samples=1):
        logits = self.model(X, samples)
        Y = tf.tile(tf.expand_dims(Y, 0), [samples, 1, 1])
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
        log_lik = -tf.reduce_sum(tf.reduce_mean(
                cross_entropy,
                axis=0),axis=0)
        return log_lik

    def KL_Divergence(self):
        """
        Should be executed after self.logLikelihood(), as it depends on weights sampled
        during the execution on model. Couldn't think of a nicer way of doing it.

        Computes the sum [over samples of w] { log(P(w|mu,sigma)) - log(P(w)) }
        """
        # If the below assertion fails, it means this function was called too early.
        # It needs to be called after the logLikelihood has been called, which samples from
        # all the weights
        assert(self.weight_samples != {})
            
        totalKL = 0
        for name, weight in self.weight_samples.items():
            # retrieve the corresponding mean and std
            mean = self.weights[name+'_mean']
            stddev = self.weights[name+'_std']
            # compute log(P(w|mu,sigma))
            logProbGivenParameters = (-0.5/stddev**2)*(weight - mean)**2
            
            # retrieve the corresponding prior
            try:
                prior_mean = self.prior[name+'_prior_mean']
                prior_stddev = self.prior[name+'_prior_std']
            except KeyError:
                prior_mean = self.default_prior_mean
                prior_stddev = self.default_prior_std
            # compute P(w)
            logProbGivenPrior = (-0.5/prior_stddev**2)*(weight - prior_mean)**2

            #TODO Sum or average over parameters?
            # presumably average over samples.
            totalKL += tf.reduce_mean(logProbGivenParameters - logProbGivenPrior)
            
        return 0



        
        


def main():
    (X, Y), (X_test, Y_test) = mnist.load_data()
    # Sort arrays
    #  indicies = np.argsort(Y)
    #  X = X[indicies]
    #  Y = Y[indicies]
    Y = tf.one_hot(Y, 10, 1.0, 0.0)
    X = tf.reshape(X, (-1,28*28))/156. - 1.
    Y_test = tf.one_hot(Y_test, 10, 1.0, 0.0)
    X_test = tf.reshape(X_test, (-1,28*28))/156. - 1.
    

    model = Gaussian_BBB_NN(num_training_samples=1, num_pred_samples=5) 
    for loss in model.train(X,Y,num_epochs=15):
        print(loss)

    testPredictions = model.predict(X_test)
    accuracy = np.sum(np.argmax(testPredictions,axis=-1)== np.argmax(Y_test.numpy(),axis=-1))/len(Y_test.numpy())
    print(accuracy)


if __name__ == "__main__":
    main()
