from datasets import Dataset, Datasets
import numpy as np


def normalize(X_n, series_length):
    X_n = X_n/(np.linalg.norm(X_n, axis=1).reshape((series_length, 1)))
    return X_n

def extract_examples(file, series_length, roi_num):
    """Extract the scans into a 3D numpy array [index, time_point, ROI_index].
    
    Args:
     f: A file object.
     
    Returns:
     data: A 3D numpy array [index, time_point, ROI_index].
    """
    
    print('Extracting', file.name)
    X = np.array([]).reshape(0, roi_num)
    X_n = []
    
    for line in file:
        # Currently there is bad data (multiple empty lines) 
        # and so check whether len(X_n)>0
        if line == '\n' and len(X_n) > 0:
            X_n = np.array(X_n).T
            X_n = normalize(X_n, series_length)
            X = np.vstack((X, X_n))
            X_n = []
        elif line != '\n':
            # Handle one scan.
            attributes = line.strip().split(",")
            X_n_row = [np.float64(attribute) for attribute in attributes[0:series_length]]
            X_n.append(X_n_row)
    
    #examples = np.array(X)
    return X

def read_data_sets(data_dir, series_length, roi_num, shuffle=True):
    """
    Read all the data into train data set.
    """
    
    with open(data_dir + '/AD.txt', 'r') as f:
        negative_examples = extract_examples(f, series_length, roi_num)
    with open(data_dir + '/Normal.txt', 'r') as f:
        positive_examples = extract_examples(f, series_length, roi_num)
    examples = np.concatenate((negative_examples, positive_examples))
    
    negative_labels = np.array([0] * (negative_examples.shape[0]/series_length))
    positive_labels = np.array([1] * (positive_examples.shape[0]/series_length))
    labels = np.concatenate((negative_labels, positive_labels))
    
    if shuffle:
        num_examples = examples.shape[0]
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        examples = examples[perm]
    
    train = Dataset(examples, labels)
    return Datasets(train=train, validation=None, test=None)

