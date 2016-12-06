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
    
    def __init__(self, input_size, state_size, label_size, dropout=True, activation=tf.nn.relu):
        self._input_size = input_size
        self._state_size = state_size
        self._label_size = label_size
        self._dropout = dropout
        self._activation = activation
    
    def dropout(self, dropout):
        self._dropout = dropout
    
    def cell_inference(self, cell_inputs, previous_state, scope=None):
        if self._dropout:
            cell_inputs = nn_ops.dropout(cell_inputs, 0.65)
        
        with variable_scope.variable_scope(scope or type(self).__name__):
            state_weights = tf.Variable(tf.random_uniform([self._state_size, self._state_size], -1.0/math.sqrt(self._state_size), 1.0/math.sqrt(self._state_size)))
            input_weights = tf.Variable(tf.random_uniform([self._input_size, self._state_size], -1.0/math.sqrt(self._input_size), 1.0/math.sqrt(self._input_size)))
            
            linear_states = math_ops.matmul(previous_state, state_weights) \
                + math_ops.matmul(cell_inputs, input_weights)
            cell_states = self._activation(linear_states)
            
            output_weights = tf.Variable(tf.random_uniform([self._state_size, self._label_size], -1.0/math.sqrt(self._state_size), 1.0/math.sqrt(self._state_size)))
            cell_outputs = math_ops.matmul(cell_states, output_weights)
            
            if self._dropout:
                cell_outputs = nn_ops.dropout(cell_outputs, 0.65)
        
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
    
    def simple_loss(self, logits, targets):
        loss = tf.reduce_mean(nn_ops.sparse_softmax_cross_entropy_with_logits(
            logits[-1], targets))
        return loss
    
    def evaluation(self, logits, labels):
        """Evaluate the quality of the logits at predicting the label.
        Args:
          logits: Logits tensor, float - [batch_size, NUM_CLASSES].
          labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).
        Returns:
          A scalar int32 tensor with the number of examples (out of batch_size)
          that were predicted correctly.
        """
        # For a classifier model, we can use the in_top_k Op.
        # It returns a bool tensor with shape [batch_size] that is true for
        # the examples where the label is in the top k (here k=1)
        # of all logits for that example.
        correct = tf.nn.in_top_k(logits[-1], labels, 1)
        # Return the number of true entries.
        return tf.reduce_sum(tf.cast(correct, tf.int32))
    
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
    

