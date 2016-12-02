import tensorflow as tf
from pygame.midi import Output
import math
import copy



class Autoencoder:
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = []
        self.biases = []
        
    def inference(self, inputs, reconstruct=True):
        """
        Build the autoencoder model up to where it may be used for inference
        """
        layer_input = inputs
        
        self.sizes.reverse()
        reverse_sizes = copy.copy(self.sizes)
        self.sizes.reverse()
        
        output_sizes = self.sizes+reverse_sizes[1:]+[int(inputs.get_shape()[1])]
        print output_sizes
        
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
        loss = tf.reduce_mean(tf.square(inputs-reconstructed_inputs))
        #loss = tf.square(inputs-reconstructed_inputs)
        return loss
    
    def train(self, loss, learning_rate):
        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        #optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss    
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
    
    def evaluation(self, encoded_inputs, inputs):
        return self.loss(encoded_inputs, inputs)

