from autoencoder import Autoencoder
from six.moves import xrange
import argparse
from autoencoder_data_reader import read_data_sets
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
                                                         FLAGS.roi_size))
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

def train(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    
    datasets = read_data_sets(FLAGS.input_data_dir)
    #data_sets = input_data.read_data_sets(FLAGS.input_data_dir)
    
    autoencoder = Autoencoder(FLAGS.layer_sizes)
    
    with tf.Graph().as_default(), tf.Session() as session:
        inputs_placeholder = placeholder_inputs(FLAGS.batch_size)
        
        encoded_inputs = autoencoder.inference(inputs_placeholder)
        inference_op = autoencoder.inference(inputs_placeholder, False)
        loss = autoencoder.loss(encoded_inputs, inputs_placeholder)
        train_op = autoencoder.train(loss, FLAGS.learning_rate)
        eval_correct = autoencoder.evaluation(encoded_inputs, inputs_placeholder)
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        checkpoint_file = os.path.join(FLAGS.log_dir, 'autoencoder.ckpt')
        session.run(init)
        
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            
            feed_dict = fill_feed_dict(datasets.train, inputs_placeholder)
            _, loss_value, encoded_inputs = session.run([train_op, loss, inference_op], feed_dict=feed_dict)
            
            duration = time.time() - start_time
            if step % 100 == 0:
                print loss_value
                print encoded_inputs[0]
                print encoded_inputs[1]
                do_eval(session, eval_correct, inputs_placeholder, datasets.train)
                saver.save(session, checkpoint_file, global_step=step)
        
        saver.save(session, checkpoint_file)
                
def inference(_):
    if tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.DeleteRecursively(FLAGS.output_dir)
    tf.gfile.MakeDirs(FLAGS.output_dir)
    
    FLAGS.batch_size = 135
    datasets = read_data_sets(FLAGS.input_data_dir, False)
    
    autoencoder = Autoencoder(FLAGS.layer_sizes)
    with tf.Graph().as_default(), tf.Session() as session:
        inputs_placeholder = placeholder_inputs(FLAGS.batch_size)
        
        inference_op = autoencoder.inference(inputs_placeholder, False)
        
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        checkpoint_file = os.path.join(FLAGS.log_dir, 'autoencoder.ckpt')
        saver.restore(session, checkpoint_file)
        
        samples_file = FLAGS.output_dir + '/encoded_samples.txt'
        while datasets.train.epochs_completed == 0:
            start_time = time.time()
            
            feed_dict = fill_feed_dict(datasets.train, inputs_placeholder)
            encoded_inputs = session.run(inference_op, feed_dict=feed_dict)
            
            with open(samples_file, 'a') as output:
                for time_series in np.transpose(encoded_inputs):
                    for value in time_series:
                        output.write(str(value) + ',')
                    output.write("\n")
                output.write("\n")
    
    labels_file = FLAGS.output_dir + '/encoded_labels.txt'
    with open(labels_file, 'a') as output:
        for label in datasets.train.labels:
            output.write(str(label) + '\n')
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--action',
        type=str,
        default='train',
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--layer_sizes',
        type=list,
        default=[200, 100, 50, 5],
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=50000,
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
        '--roi_size',
        type=int,
        default=120,
        help='Number of ROI.'
    )
    
    FLAGS, unparsed = parser.parse_known_args()
    
    if FLAGS.action == 'train':
        tf.app.run(main=train, argv=[sys.argv[0]] + unparsed)
    elif FLAGS.action == 'inference':
        tf.app.run(main=inference, argv=[sys.argv[0]] + unparsed)