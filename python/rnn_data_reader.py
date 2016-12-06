import numpy as np
from datasets import Dataset, Datasets

def normalize(X_n, time_steps):
    X_n = X_n/(np.linalg.norm(X_n, axis=1).reshape((time_steps, 1)))
    return X_n

def extract_examples(file, time_steps):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
    Args:
     f: A file object that can be passed into a gzip reader.
    Returns:
     data: A 4D unit8 numpy array [index, y, x, depth].
    Raises:
     ValueError: If the bytestream does not start with 2051.
    """
    print('Extracting', file.name)
    X = []
    X_n = []
    
    for line in file:
       if line == '\n' and len(X_n) > 0:
           X_n = np.array(X_n).T #135*5
           #X_n = normalize(X_n)
           assert X_n.shape[0] % time_steps == 0
           section_num = X_n.shape[0] / time_steps
           for split_X_n in np.split(X_n, section_num):
               split_X_n = normalize(split_X_n, time_steps)
               X.append(split_X_n)
           #X = np.vstack((X, np.array(X_n).T))
           X_n = []
       elif line != '\n':
           attributes = line.strip().split(",")
           assert len(attributes) == 136
           X_n_row = [np.float64(attribute) for attribute in attributes[0:135]]
           X_n.append(X_n_row)
    
    examples = np.array(X)
    return examples

def read_data_sets(data_dir, time_steps, shuffle=True):
    
    expanded_coeff = 135/time_steps
    with open(data_dir + '/encoded_samples.txt', 'r') as f:
        examples = extract_examples(f, time_steps)
    with open(data_dir + '/encoded_labels.txt', 'r') as f:
        labels = []
        time_series_num = 0
        for line in f:
            time_series_num += 1
            expanded_line_labels = [line.strip()] * expanded_coeff
            labels = np.concatenate((labels, expanded_line_labels))
    
    assert examples.shape[0] == labels.shape[0]
    
    #train = Dataset(examples, labels)
    train_num = int(0.9*time_series_num)*expanded_coeff
    train_examples = examples[0:train_num]
    train_labels = labels[0:train_num]
    train = Dataset(train_examples, train_labels)
    validation = Dataset(train_examples, train_labels)
    assert train_num % expanded_coeff == 0
    
    if shuffle:
        perm = np.arange(train_num)
        np.random.shuffle(perm)
        train_examples = train_examples[perm]
        train_labels = train_labels[perm]
     
#     test_num = int(0.2*time_series_num)*expanded_coeff
#     test_examples = examples[train_num:(train_num+test_num)]
#     test_labels = labels[train_num:(train_num+test_num)]
#     test = Dataset(test_examples, test_labels)
#     assert test_num % expanded_coeff == 0
#      
#     train_num = int(0.7*time_series_num)*expanded_coeff
#     validation_examples = examples[(train_num+test_num):]
#     validation_labels = labels[(train_num+test_num):]
#     validation = Dataset(validation_examples, validation_labels)
#     assert len(validation_examples) % expanded_coeff == 0
    
    return Datasets(train=train, test=None, validation=validation)