from rnn import RNN
from six.moves import xrange
import argparse
from datasets import read_data_sets
from datasets import Dataset
import tensorflow as tf
import sys
import time
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data
import os

# Basic model parameters as external flags.
FLAGS = None

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code.
  
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    inputs_placeholder: Images placeholder.
  """

  inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         FLAGS.time_steps, FLAGS.cell_legnth))
  return inputs_placeholder
  
def fill_feed_dict(dataset, inputs_placeholder):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  inputs_feed = dataset.next_batch(FLAGS.batch_size)
  
  feed_dict = {
      inputs_placeholder: inputs_feed
  }
  return feed_dict
  
def do_eval(sess,
            eval_correct,
            inputs_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
      sess: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      images_placeholder: The images placeholder.
      labels_placeholder: The labels placeholder.
      data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    # And run one epoch of eval.
    total_loss = 0.0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
      feed_dict = fill_feed_dict(data_set, inputs_placeholder)
      loss = sess.run(eval_correct, feed_dict=feed_dict) 
      total_loss += loss * int(inputs_placeholder.get_shape()[0])
    
    print "loss: " + str(total_loss/num_examples)

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    
    datasets = read_data_sets(FLAGS.input_data_dir)
    #data_sets = input_data.read_data_sets(FLAGS.input_data_dir)
    
    rnn = RNN()
    
    with tf.Graph().as_default(), tf.Session() as session:
        inputs_placeholder = tf.placeholder(tf.int32, 
            shape=(batch_size, num_steps))
        initial_state = tf.zeros([batch_size, state_size], tf.float32)
        
        rnn = RNN(state_size, vocab_size, embedding_size)
        embedding_inputs = rnn.embeddings(inputs_placeholder)
        
        outputs = []
        state = initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_state, cell_output = rnn.cell_inference(embedding_inputs[:, time_step, :], state)
                outputs.append(cell_output)
                
        y_loss = tf.placeholder(tf.int32, [batch_size, num_steps])
        y_eval = tf.placeholder(tf.int32, [batch_size, num_steps])
        #logits = tf.reshape(tf.concat(1, outputs), [batch_size, num_steps, vocab_size])
        loss = rnn.loss(outputs, y_loss)
        corrects = rnn.evaluation(outputs, y_eval)
        
        train_op = rnn.training(loss, learning_rate)
        
        tf.initialize_all_variables().run()
        
        for i in range(max_epoch):
            for step, (x, y) in enumerate(reader.ptb_iterator(train_data, batch_size,
                                            num_steps)):
                print "epoch: " + str(step/num_steps) + ";step: " + str(step % num_steps)
                cost, correct_num, _ = session.run([loss, corrects, train_op],
                                {inputs_placeholder: x,
                                 y_loss: y,
                                 y_eval: y,
                                 state: initial_state.eval()})
                print "correct_num: " + str(correct_num)
                

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--time_steps',
        type=int,
        default=9,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
      '--batch_size',
      type=int,
      default=135,
      help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/tmp/data',
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default='/tmp/data',
        help='Directory to put the input data.'
    )
    parser.add_argument(
      '--log_dir',
      type=str,
      default='/tmp/data',
      help='Directory to put the log data.'
  )
    parser.add_argument(
        '--cell_length',
        type=int,
        default=5,
        help='Number of ROI.'
    )
    
    FLAGS, unparsed = parser.parse_known_args()
    
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
