import numpy as np

def normalize(X_n):
    X_n = X_n/(np.linalg.norm(X_n, axis=1).reshape((135, 1)))
    return X_n

def extract_examples(file, cell_length, time_steps):
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
           assert X_n.shape[1] == cell_length
           assert X_n.shape[0] % time_steps == 0
           section_num = X_n.shape[0] / time_steps
           X.append(np.split(X_n, section_num))
           #X = np.vstack((X, np.array(X_n).T))
           X_n = []
       elif line != '\n':
           attributes = line.strip().split(",")
           assert len(attributes) == 136
           X_n_row = [np.float64(attribute) for attribute in attributes[0:135]]
           X_n.append(X_n_row)
    
    examples = np.array(X)
    return examples

def read_data_sets(data_dir, cell_length, time_steps, shuffle=True):
    
    with open(data_dir + '/encoded_samples.txt', 'r') as f:
        examples = extract_examples(f, cell_length, time_steps)
    with open(data_dir + '/encoded_labels.txt', 'r') as f:
        labels = []
        for line in f:
            labels.append([line.strip()] * (135/time_steps))
    
    if shuffle:
        num_examples = examples.shape[0]
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        examples = examples[perm]
        labels = examples[perm]
    
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