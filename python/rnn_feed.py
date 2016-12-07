from rnn import RNN
from six.moves import xrange
import argparse
from rnn_data_reader import read_data_sets
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
                                                         FLAGS.time_steps, FLAGS.input_size))
  return inputs_placeholder
  
def fill_feed_dict(dataset, inputs_placeholder, labels_placeholder, shuffle=True):
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
  samples, labels = dataset.next_batch_examples(FLAGS.batch_size, shuffle)
  
  feed_dict = {
      inputs_placeholder: samples,
      labels_placeholder: labels
  }
  return feed_dict
  
def do_eval(sess,
            eval_correct,
            inputs_placeholder,
            labels_placeholder,
            data_set,
            num_examples_coeff=1):
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
    true_count = 0  # Counts the number of correct predictions.  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * (FLAGS.batch_size/num_examples_coeff)
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, inputs_placeholder, labels_placeholder, False)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    
    input_size = FLAGS.input_size
    state_size = FLAGS.state_size
    label_size = FLAGS.label_size
    batch_size = FLAGS.batch_size
    time_steps = FLAGS.time_steps
    
    assert(batch_size % (120/time_steps) == 0)
    
    datasets = read_data_sets(FLAGS.input_data_dir, time_steps)
    #data_sets = input_data.read_data_sets(FLAGS.input_data_dir)
    with tf.Graph().as_default(), tf.Session() as session:
        inputs_placeholder = placeholder_inputs(FLAGS.batch_size)
        labels_placeholder = tf.placeholder(tf.int32, [time_steps, batch_size])
        eval_labels_placeholder = tf.placeholder(tf.int32, [batch_size])
        
        rnn = RNN(input_size, state_size, label_size)
        
        with tf.variable_scope("RNN"):
            train_logits = rnn.reference(inputs_placeholder, dropout=True)
            test_logits = rnn.reference(inputs_placeholder)
        #loss = rnn.loss(logits, eval_labels_placeholder)
        loss = rnn.simple_loss(train_logits, eval_labels_placeholder)
        train_op = rnn.train(loss, FLAGS.learning_rate)

        eval_correct = rnn.evaluation(test_logits, eval_labels_placeholder)
        subjects_eval_correct = rnn.subjects_evaluation(test_logits, eval_labels_placeholder, time_steps)
        
        init = tf.global_variables_initializer()
        session.run(init)
        
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            
            feed_dict = fill_feed_dict(datasets.train, inputs_placeholder, eval_labels_placeholder)
            _, loss_value= session.run([train_op, loss], feed_dict=feed_dict)
            
            if step % 1000 == 0:
                print loss_value
            if (step + 1) % 5000 == 0 or (step + 1) == FLAGS.max_steps:
                print('Training Data Eval:')
                do_eval(session,
                        eval_correct,
                        inputs_placeholder,
                        eval_labels_placeholder,
                        datasets.train)
                 
                print('Training Subjects Data Eval:')
                do_eval(session,
                        subjects_eval_correct,
                        inputs_placeholder,
                        eval_labels_placeholder,
                        datasets.validation,
                        120/time_steps)
                 
                print('Test Data Eval:')
                do_eval(session,
                        eval_correct,
                        inputs_placeholder,
                        eval_labels_placeholder,
                        datasets.test)
                
                print('Test Subjects Data Eval:')
                do_eval(session,
                        subjects_eval_correct,
                        inputs_placeholder,
                        eval_labels_placeholder,
                        datasets.test,
                        120/time_steps)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=200000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--time_steps',
        type=int,
        default=12,
        help='number of time steps in a series'
    )
    parser.add_argument(
      '--batch_size',
      type=int,
      default=10,
      help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--input_size',
        type=int,
        default=5,
        help='vector length of the hidden state'
    )
    parser.add_argument(
        '--state_size',
        type=int,
        default=40,
        help='vector length of the hidden state'
    )
    parser.add_argument(
        '--label_size',
        type=int,
        default=2,
        help='number of classes'
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
    
    FLAGS, unparsed = parser.parse_known_args()
    
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
