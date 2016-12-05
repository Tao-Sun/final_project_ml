import math
import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

flags = tf.flags
flags.DEFINE_string("data_path", None, "data_path")
FLAGS = flags.FLAGS

class RNN(object):
    
    def __init__(self, input_size, state_size, label_size, activation=tanh):
        self._input_size = input_size
        self._state_size = state_size
        self._label_size = label_size
        self._activation = activation
    
    def cell_inference(self, cell_inputs, previous_state, scope=None):
        with variable_scope.variable_scope(scope or type(self).__name__):
            state_weights = tf.Variable(tf.random_uniform([self._state_size, self._state_size], -1.0/math.sqrt(self._state_size), 1.0/math.sqrt(self._state_size)))
            input_weights = tf.Variable(tf.random_uniform([self._input_size, self._state_size], -1.0/math.sqrt(self._input_size), 1.0/math.sqrt(self._input_size)))
            
            linear_states = math_ops.matmul(previous_state, state_weights) \
                + math_ops.matmul(cell_inputs, input_weights)
            cell_states = self._activation(linear_states)
            
            output_weights = tf.Variable(tf.random_uniform([self._state_size, self._label_size], -1.0/math.sqrt(self._state_size), 1.0/math.sqrt(self._state_size)))
            cell_outputs = math_ops.matmul(cell_states, output_weights)
        
        return cell_states, cell_outputs
    
    def loss(self, logits, targets):
        time_steps = len(logits)
        
        crossents = []
        for time_step in range(time_steps):
            logit = logits[time_step]
            target = targets[time_step]
            crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                logit, target)
            crossents.append(crossent)
          
        loss = tf.reduce_mean(math_ops.add_n(crossents))
#         loss = tf.reduce_mean(nn_ops.sparse_softmax_cross_entropy_with_logits(
#             logits[-1], targets))
        return loss
    
    def evaluation(self, logits, targets):
        correct_num = 0
        for num_step in range(35):
            logit = logits[num_step]
            target = targets[:, num_step]
            correct = tf.nn.in_top_k(logit, target, 1)
            correct_num += tf.reduce_sum(tf.cast(correct, tf.int32))
        return correct_num
    
    def train(self, loss, learning_rate):
        # Create the gradient descent optimizer with the given learning rate.
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss    
        train_op = optimizer.minimize(loss)
        return train_op
     
# def main(_):
#     raw_data = reader.ptb_raw_data(FLAGS.data_path)
#     train_data, valid_data, test_data, _ = raw_data
#     
#     batch_size = 20
#     state_size = 200
#     input_size= 600
#     label_size = 10000
#     learning_rate = 1.0 
#     max_epoch = 39
#     num_steps = 35
#     
#     with tf.Graph().as_default(), tf.Session() as session:
#         inputs_placeholder = tf.placeholder(tf.int32, 
#             shape=(batch_size, num_steps))
#         initial_state = tf.zeros([batch_size, state_size], tf.float32)
#         
#         rnn = RNN(state_size, label_size, input_size)
#         embedding_inputs = rnn.embeddings(inputs_placeholder)
#         
#         outputs = []
#         state = initial_state
#         with tf.variable_scope("RNN"):
#             for time_step in range(num_steps):
#                 if time_step > 0: tf.get_variable_scope().reuse_variables()
#                 cell_state, cell_output = rnn.cell_inference(embedding_inputs[:, time_step, :], state)
#                 outputs.append(cell_output)
#                 
#         y_loss = tf.placeholder(tf.int32, [batch_size, num_steps])
#         y_eval = tf.placeholder(tf.int32, [batch_size, num_steps])
#         #logits = tf.reshape(tf.concat(1, outputs), [batch_size, num_steps, label_size])
#         loss = rnn.loss(outputs, y_loss)
#         corrects = rnn.evaluation(outputs, y_eval)
#         
#         train_op = rnn.training(loss, learning_rate)
#         
#         tf.initialize_all_variables().run()
#         
#         for i in range(max_epoch):
#             for step, (x, y) in enumerate(reader.ptb_iterator(train_data, batch_size,
#                                             num_steps)):
#                 print "epoch: " + str(step/num_steps) + ";step: " + str(step % num_steps)
#                 cost, correct_num, _ = session.run([loss, corrects, train_op],
#                                 {inputs_placeholder: x,
#                                  y_loss: y,
#                                  y_eval: y,
#                                  state: initial_state.eval()})
#                 print "correct_num: " + str(correct_num)
# 
# 
# if __name__ == '__main__':
#     tf.app.run()       
    

