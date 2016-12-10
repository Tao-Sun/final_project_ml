"""Builds the autoencoder network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the autoencoder
reconstruct the inputs.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to optimize the model.

This file is used by the "autoencoder_feed.py" file and not meant to
be run.
"""

import copy
import math

from pygame.midi import Output

import tensorflow as tf


class Autoencoder:
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = []
        self.biases = []
        
    def inference(self, inputs, reconstruct=True):
        """
        Build the autoencoder model up to where it may be used for reconstructing inputs.
        
        Args:
            inputs: time points placeholder, [batch_size, ROI_num]
            reconstruct: Whether to reconstruct the inputs or just encode the inputs
                         to a lower dimension.
            
        Returns: Reconstructed inputs or encoded inputs.
        """
        
        layer_input = inputs
        
        self.sizes.reverse()
        reverse_sizes = copy.copy(self.sizes)
        self.sizes.reverse()
        
        output_sizes = self.sizes+reverse_sizes[1:]+[int(inputs.get_shape()[1])]
        print "Layers configuration: " + str(output_sizes)
        
        for i, output_size in enumerate(output_sizes):
            input_size = int(layer_input.get_shape()[1])
            
            weights = tf.Variable(tf.random_uniform([input_size, output_size], -1.0/math.sqrt(input_size), 1.0/math.sqrt(input_size)))
            biases = tf.Variable(tf.zeros([output_size]))
            self.weights.append(weights)
            self.biases.append(biases)
            
            outputs = tf.nn.tanh(tf.add(tf.matmul(layer_input, weights), biases))
            if reconstruct == False and i == len(self.sizes)-1:
                return outputs
                
            layer_input = outputs
            
        reconstruct_inputs = outputs
        return reconstruct_inputs
    
    def loss(self, reconstructed_inputs, inputs):
        """Calculates mean squared difference betwen reconstructed_inputs and inputs of the batch.
        
        Args:
          reconstructed_inputs: reconstructed inputs from autoencoder, float - [batch_size, ROI_num].
          inputs: inputs of the autoencoder, float - [batch_size, ROI_num].
          
        Returns:
          loss: Loss tensor of type float.
        """
        loss = tf.reduce_mean(tf.square(inputs-reconstructed_inputs))
        return loss
    
    def train(self, loss, learning_rate):
        """Sets up the training Ops.
        
        Creates an Adadelta optimizer.
        
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.
        
        Args:
          loss: Loss tensor, from loss().
          learning_rate: The learning rate to use for the optimizer.
          
        Returns:
          train_op: The Op for training.
        """
        
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
    
    def evaluation(self, encoded_inputs, inputs):
        """For evaluation purpose of the model; just return loss of the batch.
        
        Args:
          reconstructed_inputs: reconstructed inputs from autoencoder, float - [batch_size, ROI_num].
          inputs: inputs of the autoencoder, float - [batch_size, ROI_num].
          
        Returns:
          loss: Loss tensor of type float.
        """
        return self.loss(encoded_inputs, inputs)

