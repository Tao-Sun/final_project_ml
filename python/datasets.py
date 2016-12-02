import numpy as np
import collections

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

def normalize(X_n):
    X_n = X_n/(np.linalg.norm(X_n, axis=1).reshape((135, 1)))
    return X_n

def extract_examples(file):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
    Args:
     f: A file object that can be passed into a gzip reader.
    Returns:
     data: A 4D unit8 numpy array [index, y, x, depth].
    Raises:
     ValueError: If the bytestream does not start with 2051.
    """
    print('Extracting', file.name)
    X = np.array([]).reshape(0, 120)
    X_n = []
    
    for line in file:
       if line == '\n' and len(X_n) > 0:
           X_n = np.array(X_n).T
           X_n = normalize(X_n)
           assert X_n.shape == (135, 120)
           X = np.vstack((X, X_n))
           #X = np.vstack((X, np.array(X_n).T))
           X_n = []
       elif line != '\n':
           attributes = line.strip().split(",")
           assert len(attributes) == 136
           X_n_row = [np.float64(attribute) for attribute in attributes[0:135]]
           X_n.append(X_n_row)
    
    #examples = np.array(X)
    return X

class Dataset(object):
    def __init__(self, examples, labels):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
            `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        #assert images.shape[0] == labels.shape[0]
        self._num_examples = examples.shape[0]
        self._examples = examples
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def examples(self):
        return self._examples
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
    
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._examples = self._examples[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        elif self._index_in_epoch == self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
        end = self._index_in_epoch
        return self._examples[start:end]

def read_data_sets(data_dir, shuffle=True):
    
    with open(data_dir + '/AD.txt', 'r') as f:
        negative_examples = extract_examples(f)
    with open(data_dir + '/Normal.txt', 'r') as f:
        positive_examples = extract_examples(f)
    examples = np.concatenate((negative_examples, positive_examples))
    
    assert negative_examples.shape[0] % 135 == 0
    assert positive_examples.shape[0] % 135 == 0
    negative_labels = np.array([-1] * (negative_examples.shape[0]/135))
    positive_labels = np.array([1] * (positive_examples.shape[0]/135))
    labels = np.concatenate((negative_labels, positive_labels))
    
    if shuffle:
        num_examples = examples.shape[0]
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        examples = examples[perm]
    
    train = Dataset(examples, labels)
    
#     train_examples = examples[0:int(0.7*num_examples)]
#     train_labels = labels[0:int(0.7*num_examples)]
#     train = Dataset(train_examples, train_labels)
#     
#     test_examples = examples[int(0.7*num_examples):int(0.9*num_examples)]
#     test_labels = labels[int(0.7*num_examples):int(0.9*num_examples)]
#     test = Dataset(test_examples, test_labels)
#     
#     validation_examples = examples[int(0.9*num_examples):]
#     validation_labels = examples[int(0.9*num_examples):]
#     validation = Dataset(validation_examples, validation_labels)
    
    return Datasets(train=train, validation=None, test=None)