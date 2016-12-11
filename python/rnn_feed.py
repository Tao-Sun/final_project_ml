import argparse
from matplotlib.legend_handler import HandlerLine2D
import os
import random
import sys
import time

from datasets import Dataset
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from rnn import RNN
from rnn_data_reader import read_data_sets
from six.moves import xrange
import tensorflow as tf


# Basic model parameters as external flags.
FLAGS = None

def placeholder_inputs(batch_size):
    """"Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code.
    
    Args:
      batch_size: The batch size will be baked into both placeholders.
    Returns:
      inputs_placeholder: placeholder for time points in a sub-series sequence
      labels_placeholder: labels placeholder
    """
    
    inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           FLAGS.time_steps, FLAGS.input_size))
    labels_placeholder = tf.placeholder(tf.int32, [batch_size])
    
    return inputs_placeholder, labels_placeholder
  
def fill_feed_dict(dataset, inputs_placeholder, labels_placeholder, shuffle=True):
  """Fills the feed_dict for training the given step.
  
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  
  Args:
    data_set: The set of time points, from autoencoder_data_reader.read_data_sets().
    inputs_placeholder: The time points placeholder, from placeholder_inputs().
    labels_placeholder: The labels placeholder, from placeholder_inputs().
    
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
        inputs_placeholder: placeholder for time points in a sub-series sequenc
        labels_placeholder: placeholder for labels
        data_set: The set of images and labels to evaluate, from
          rnn_data_reader.read_data_sets().
    
    Returns:
        precesion: precesion of the evaluation.
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * (FLAGS.batch_size/num_examples_coeff)
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, inputs_placeholder, labels_placeholder, False)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    
#     print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
#         (num_examples, true_count, precision))
    
    return precision

def output_data(train_precision, 
                train_subject_precision, 
                test_precision, 
                test_subject_precision,
                file_time):
    eval_file = FLAGS.output_dir + '/eval_' + str(file_time) + '.txt'
    with open(eval_file, 'a') as output:
        output.write(str(FLAGS) + '\n')
        output.write(str(train_precision) + '\n')
        output.write(str(train_subject_precision) + '\n')
        output.write(str(test_precision) + '\n')
        output.write(str(test_subject_precision) + '\n')

def plot_data(train_precision, 
              train_subject_precision, 
              test_precision, 
              test_subject_precision,
              file_time):
    plot_file = FLAGS.output_dir + '/plot_' + str(file_time) + '.png'
    
    plt.figure(figsize=(20, 9))
    ax = plt.gca()
    
    xaxisLocator = ticker.MultipleLocator(base=10000)
    ax.yaxis.set_major_locator(xaxisLocator)
    
    ax.set_ylim([0, 1])
    yaxisLocator = ticker.MultipleLocator(base=0.05)
    ax.yaxis.set_major_locator(yaxisLocator)
    
    ax.set_title('Classification Accuracy With RNN Classifiers and the Ensemble Learner')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Accuracy')
    
    red_line = mlines.Line2D([], [], color='red', marker='o', markersize=5)
    green_line = mlines.Line2D([], [], color='green', marker='o', markersize=5)
    blue_line = mlines.Line2D([], [], color='blue', marker='o', markersize=5)
    cyan_line = mlines.Line2D([], [], color='cyan', marker='o', markersize=5)
    handler_map = {red_line: HandlerLine2D(numpoints=1),
                   green_line: HandlerLine2D(numpoints=1),
                   blue_line: HandlerLine2D(numpoints=1), 
                   cyan_line: HandlerLine2D(numpoints=1),
                  }
    plt.legend([red_line, green_line, blue_line, cyan_line], ["Train", "Train in ensemble", "Test", "Test in ensemble"], handler_map=handler_map, loc=4)
    
    plot_range = range(0, np.multiply(1000, len(train_precision)), 1000)
    plt.plot(plot_range, np.array(train_precision), 'ro-')
    plt.plot(plot_range, np.array(train_subject_precision), 'go-')
    plt.plot(plot_range, np.array(test_precision), 'bo-')
    plt.plot(plot_range, np.array(test_subject_precision), 'co-')
    
    plt.savefig(plot_file)

def main(_):
    if tf.gfile.Exists(FLAGS.output_dir) != True:
        tf.gfile.MakeDirs(FLAGS.output_dir)
    
    input_size = FLAGS.input_size
    state_size = FLAGS.state_size
    label_size = FLAGS.label_size
    batch_size = FLAGS.batch_size
    time_steps = FLAGS.time_steps
    series_length = FLAGS.series_length
    
    if (batch_size*time_steps) % series_length != 0:
        print batch_size
        print time_steps
        print series_length
        raise ValueError('(batch_size*time_steps)%series_length shoule equal 0')
    
    datasets = read_data_sets(FLAGS.input_data_dir, series_length, time_steps, input_size)
    #data_sets = input_data.read_data_sets(FLAGS.input_data_dir)
    with tf.Graph().as_default(), tf.Session() as session:
        inputs_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
        
        rnn = RNN(input_size, state_size, label_size)
        
        with tf.variable_scope("RNN"):
            train_logits = rnn.reference(inputs_placeholder, dropout=True)
            test_logits = rnn.reference(inputs_placeholder)
        #loss = rnn.loss(logits, labels_placeholder)
        loss = rnn.simple_loss(train_logits, labels_placeholder)
        train_op = rnn.train(loss, FLAGS.learning_rate)

        eval_correct = rnn.evaluation(test_logits, labels_placeholder)
        subjects_eval_correct = rnn.subjects_evaluation(test_logits, 
                                                        labels_placeholder, 
                                                        series_length,
                                                        time_steps)
        
        init = tf.global_variables_initializer()
        session.run(init)
        
        train_precision = [0]
        train_subject_precision = [0]
        test_precision = [0]
        test_subject_precision = [0]
        
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            
            feed_dict = fill_feed_dict(datasets.train, inputs_placeholder, labels_placeholder)
            _, loss_value= session.run([train_op, loss], feed_dict=feed_dict)
            
#             if (step + 1) % 100 == 0:
#                 print loss_value
            if (step + 1) % 10000 == 0:
                print str(step+1) + " steps completed!"
#             if (step + 1) % 10000 == 0:
#                 print train_precision
#                 print train_subject_precision
#                 print test_precision
#                 print test_subject_precision
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                #print('Training Data Eval:')
                train_precision.append(do_eval(session,
                                               eval_correct,
                                               inputs_placeholder,
                                               labels_placeholder,
                                               datasets.train))
                 
                #print('Training Subjects Data Eval:')
                train_subject_precision.append(do_eval(session,
                                               subjects_eval_correct,
                                               inputs_placeholder,
                                               labels_placeholder,
                                               datasets.validation,
                                               series_length/time_steps))
                 
                #print('Test Data Eval:')
                test_precision.append(do_eval(session,
                        eval_correct,
                        inputs_placeholder,
                        labels_placeholder,
                        datasets.test))
                
                #print('Test Subjects Data Eval:')
                test_subject_precision.append(do_eval(session,
                        subjects_eval_correct,
                        inputs_placeholder,
                        labels_placeholder,
                        datasets.test,
                        series_length/time_steps))
        
        file_time = time.time()
        output_data(train_precision, 
                    train_subject_precision, 
                    test_precision, 
                    test_subject_precision,
                    file_time)
        
        plot_data(train_precision, 
                  train_subject_precision, 
                  test_precision, 
                  test_subject_precision,
                  file_time)

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
        '--series_length',
        type=int,
        default=135,
        help='Length of whole time series'
    )
    parser.add_argument(
        '--time_steps',
        type=int,
        default=9,
        help='number of time steps in a series'
    )
    parser.add_argument(
      '--batch_size',
      type=int,
      default=15,
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
        default=80,
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
