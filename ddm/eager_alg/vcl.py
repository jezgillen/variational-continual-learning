#!/usr/bin/python3

import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from sklearn import metrics
import pickle
import copy

tf.enable_eager_execution()
tf.set_random_seed(4)

class Neural_Net_Base():
    """
    A neural net base class, containing functions for training
    """
    def __init__(self):
        self.opt = None
        self.weights = {}
        self.checkpoint = None

    def save(self, filename):
        # TODO rewrite to use tensorflow's normal checkpointing system
        saved_weights = {}
        for name, weight in self.weights.items():
            saved_weights[name] = copy.deepcopy(weight.numpy())
        with open(filename, 'wb') as f:
            pickle.dump(saved_weights, f)

    def load(self, path):
        """
        This function appears to work when used across different runs of the program
        However, when used in the same session, it doesn't work.
        Possibly a caching issue?
        """
        self.weights = {}
        with open(path, 'rb') as f:
            saved_weights = pickle.load(f)
        for name, weight in saved_weights.items():
            try:
                tf.assign(self.weights[name], weight)
            except:
                self.weights[name] = tf.Variable(initial_value=weight)


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
           minimizing, such as comparing multiple models, then I might need to add in constants
           to the logLiklihood and KLdivergence calculations
         - Try bimodal prior and posterior
    """
    def __init__(self, num_training_samples=7, num_pred_samples=99,
                optimizer=None):

       super().__init__()
       self.num_training_samples = num_training_samples
       self.num_pred_samples = num_pred_samples
       self.opt = optimizer 
       self.default_prior_mean = 0.
       self.default_prior_std = 1.0
       self.weight_samples = {}
       self.prior = {}

    def set_prior(self, weights):
        for name, weight in weights.items():
            self.prior[name] = tf.Variable(weight,trainable=False)

    def get_posterior(self):
        return self.weights

    def predict(self, X):
        logits = self.model(X, samples=self.num_pred_samples,training=False)
        predictions = tf.nn.softmax(logits)
        return tf.reduce_mean(predictions, axis=0).numpy()

    def model(self, X, samples=1,training=True):
        """
        Returns logit outputs of the model, with shape [sample, batch, logit]
        """

        # Transforms inputs from [batch, input] into [sample, batch, input]
        X = tf.tile(tf.expand_dims(X, 0), [samples, 1, 1])        

        z = self.fully_connected_layer(X, samples, size=(28*28, 128), name="layer_one")
        h = tf.nn.relu(z)
        tf.layers.dropout(h, rate=0.8)
        z = self.fully_connected_layer(h, samples, size=(128, 64), name="layer_two")
        h = tf.nn.relu(z)
        tf.layers.dropout(h, rate=0.8)
        z = self.fully_connected_layer(h, samples, size=(64, 10), name="layer_three")
        logits = z
        # logits should still be of shape [sample, batch, logit]
        return logits

    def fully_connected_layer(self, x, samples, size, name):
        try:
            weights_mean = self.weights[name+'_weights_mean']
            weights_logvar = self.weights[name+'_weights_logvar']
            bias_mean = self.weights[name+'_bias_mean']
            bias_logvar = self.weights[name+'_bias_logvar']
        except KeyError:
            weights_mean = self.weights[name+'_weights_mean'] = tf.Variable(tf.zeros(size))
            weights_logvar = self.weights[name+'_weights_logvar'] = \
                                        tf.Variable(tf.ones(size)*-0.6)
            bias_mean = self.weights[name+'_bias_mean'] = tf.Variable(tf.zeros(size[-1]))
            bias_logvar = self.weights[name+'_bias_logvar'] = \
                                        tf.Variable(tf.ones(size[-1])*-0.6)


        weights_sample = tf.random_normal((samples, *size), 
                                          mean=weights_mean,
                                          stddev=tf.sqrt(tf.exp(weights_logvar*10)))
        self.weight_samples[name+'_weights'] = weights_sample

        bias_sample = tf.random_normal((samples, x.shape[-2], size[-1]), 
                                          mean=bias_mean,
                                          stddev=tf.sqrt(tf.exp(bias_logvar*10)))
        self.weight_samples[name+'_bias'] = bias_sample
        return x@weights_sample + bias_sample

    def loss(self, X, Y, samples=None):
        if samples is None:
            samples = self.num_training_samples
        logLikelihood = self.logLikelihood(X, Y, samples)
        # I don't at all understand why dividing by the number of training samples seems to give
        # the best results, but it seems to. TODO figure this out later
        KL_Divergence = self.KL_Divergence()/60#/tf.cast(tf.shape(X),tf.float32)[0]
        return -logLikelihood + KL_Divergence

    def logLikelihood(self, X, Y, samples=1):
        logits = self.model(X, samples)
        Y = tf.tile(tf.expand_dims(Y, 0), [samples, 1, 1])
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
        # cross entropy is of shape [sample, batch]
        # average across samples, but sum across examples, see self.loss() for more info
        log_lik = -tf.reduce_sum(tf.reduce_mean(cross_entropy,axis=0),axis=0)
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
        num_weights = 0
        for name, weight in self.weight_samples.items():
            num_weights += 1

            # retrieve the corresponding mean and logvar
            mean = self.weights[name+'_mean']
            logvar = self.weights[name+'_logvar']
            # compute log(P(w|mu,sigma))
            logProbGivenParameters = (-0.5/tf.exp(logvar*10))*(weight - mean)**2
            
            # retrieve the corresponding prior
            try:
                prior_mean = self.prior[name+'_mean']
                prior_logvar = self.prior[name+'_logvar']
            except KeyError:
                prior_mean = self.default_prior_mean
                prior_logvar = tf.log(self.default_prior_std**2)
            # compute P(w)
            logProbGivenPrior = (-0.5/tf.exp(prior_logvar))*(weight - prior_mean)**2

            # Sum over parameters?
            # average over samples.
            KL = logProbGivenParameters - logProbGivenPrior
            totalKL += tf.reduce_sum(tf.reduce_mean(KL, axis=0))
            
        return totalKL

class Reparameterised_Gaussian_BBB_NN(Gaussian_BBB_NN):
    """
    A modified version of the Gaussian Bayes By Backprop neral net above, 
    using the reparameterisation trick described in 
    "Variational Dropout and the Local Reparameterization Trick" 
    [https://arxiv.org/pdf/1506.02557.pdf]

    This reduces the variance of the weight updates, but slows down computation slightly.
    """
    def model(self, X, samples=1,training=True):
        """
        Returns logit outputs of the model, with shape [sample, batch, logit]
        """
        batch_size = X.shape[0]

        # Transforms inputs from [batch, input] into [sample, batch, input]
        X = tf.tile(tf.expand_dims(X, 0), [samples, 1, 1])        

        # Reshape to [sample, batch, height, width, channels]
        #  X = tf.reshape(X, (samples, batch_size, 28, 28, 1))
        #  z = self.convolutional_layer(X, (5,5,1,4), name="conv_layer",strides=[1,1,1,1])
        #  h = tf.nn.relu(z)
        #  z = self.convolutional_layer(X, (3,3,1,4), name="conv_layer",strides=[1,1,1,1])
        #  h = tf.nn.relu(z)
        #  z = self.convolutional_layer(X, (3,3,1,4), name="conv_layer",strides=[1,1,1,1])
        #  z = tf.reshape(z, (samples, batch_size, -1))
        z = self.reparameterised_fully_connected_layer(X, samples, size=(28*28, 128), name="layer_one")
        h = tf.nn.relu(z)
        z = self.reparameterised_fully_connected_layer(h, samples, size=(h.shape[-1], 64), name="layer_two")
        h = tf.nn.relu(z)
        z = self.reparameterised_fully_connected_layer(h, samples, size=(64, 10), name="layer_three")
        logits = z
        # logits should still be of shape [sample, batch, logit]
        return logits

    def convolutional_layer(self, x, kernel_size, name, strides=[1,1,1,1], padding="VALID"):
        # x is of shape [sample, batch, height, width, channels]
        samples = x.shape[0]
        batch_size = x.shape[1]
        x = tf.reshape(x, (-1,*x.shape[-3:]))
        # x is now of shape [sample*batch, height, width, channels]

        #retrieve kernel
        try:
            weights_mean = self.weights[name+'_weights_mean']
            weights_logvar = self.weights[name+'_weights_logvar']
        except KeyError:
            weights_mean = self.weights[name+'_weights_mean'] = tf.Variable(tf.zeros(kernel_size))
            weights_logvar = self.weights[name+'_weights_logvar'] = tf.Variable(tf.ones(kernel_size)*-6)

        output_mean = tf.nn.conv2d(x, weights_mean, strides=strides, padding=padding)
        # This diverges a bit from the Baysian CNN paper because I don't understand why they did their thing
        output_std = tf.sqrt(1e-8 + tf.nn.conv2d(x**2, tf.exp(weights_logvar), strides=strides, padding=padding))

        output_mean = tf.reshape(output_mean, (samples, batch_size, *output_mean.shape[-3:]))
        output_std = tf.reshape(output_std, (samples, batch_size, *output_mean.shape[-3:]))
        output_sample = tf.random_normal(output_mean.shape,
                                         mean=output_mean,
                                         stddev=output_std)

        return output_sample 

    def reparameterised_fully_connected_layer(self, x, samples, size, name):
        try:
            weights_mean = self.weights[name+'_weights_mean']
            weights_logvar = self.weights[name+'_weights_logvar']
            bias_mean = self.weights[name+'_bias_mean']
            bias_logvar = self.weights[name+'_bias_logvar']
        except KeyError:
            weights_mean = self.weights[name+'_weights_mean'] = tf.Variable(tf.zeros(size))
            weights_logvar = self.weights[name+'_weights_logvar'] = tf.Variable(tf.ones(size)*-6)
            bias_mean = self.weights[name+'_bias_mean'] = tf.Variable(tf.zeros(size[-1]))
            bias_logvar = self.weights[name+'_bias_logvar'] = tf.Variable(tf.ones(size[-1])*-6)


        batch_size = x.shape[-2]
        x = tf.reshape(x, (-1, size[0]))
        
        output_mean = x@weights_mean + bias_mean
        output_std = tf.sqrt((x**2)@(tf.exp(weights_logvar)) + tf.exp(bias_logvar))
        # both of shape [batch*samples, output_size]

        output_mean = tf.reshape(output_mean, (samples, batch_size, size[1]))
        output_std = tf.reshape(output_std, (samples, batch_size, size[1]))
        output_sample = tf.random_normal((samples, batch_size, size[1]),
                                         mean=output_mean,
                                         stddev=output_std)
        return output_sample

    def KL_Divergence(self):
        totalKL = 0
        for name, weight in self.weights.items():
            if(not name.endswith("_mean")):
                continue
            name = "_".join(name.split("_")[:-1])

            mean = weight
            logvar = self.weights[name+"_logvar"]
            try:
                prior_mean = self.prior[name+'_mean']
                prior_logvar = self.prior[name+'_logvar']
            except KeyError:
                prior_mean = self.default_prior_mean
                prior_logvar = tf.log(self.default_prior_std**2)
            
            # This should work as long as samples is in axis 0 and weights are everything else
            totalKL += 0.5*tf.reduce_sum(tf.reduce_mean(tf.log(tf.exp(prior_logvar)/tf.exp(logvar)) - 1 + ((mean - prior_mean)**2)/tf.exp(prior_logvar) + tf.exp(logvar)/tf.exp(prior_logvar),axis=0)) 
        return totalKL



def main():
    (X, Y), (X_test, Y_test) = mnist.load_data()

    Y_test = tf.one_hot(Y_test, 10, 1.0, 0.0)
    X_test = tf.reshape(X_test, (-1,28*28))/156.
    

    #  model = Gaussian_BBB_NN(num_training_samples=1, num_pred_samples=5) 
    model = Reparameterised_Gaussian_BBB_NN(num_training_samples=1, num_pred_samples=5) 
    for i in range(5):
        print(f"\n\nTraining on data with label: {i*2} and {i*2+1}")
        _Y = Y[Y<i*2+2]
        _X = X[Y<i*2+2]
        _Y = tf.one_hot(_Y, 10, 1.0, 0.0)
        _X = tf.reshape(_X, (-1,28*28))/156.
        
        for loss in model.train(_X,_Y,num_epochs=1,batch_size=64):
            print(f"Loss: {loss}")

            testPredictions = model.predict(X_test)
            y_true = np.argmax(Y_test.numpy(),axis=-1)
            y_pred = np.argmax(testPredictions,axis=-1)
            accuracy = np.sum(y_pred == y_true)/len(Y_test.numpy())
            print(f"Test Accuracy: {accuracy*100:.2F}%")
        print(metrics.confusion_matrix(y_true, y_pred))

        posterior = model.get_posterior()
        model.set_prior(posterior)

    model.save("trained_separately")


if __name__ == "__main__":
    main()
