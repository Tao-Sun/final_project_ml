import numpy as np
from datasets import Dataset, Datasets

def normalize(X_n, time_steps):
    X_n = X_n/(np.linalg.norm(X_n, axis=1).reshape((time_steps, 1)))
    return X_n

def extract_subject_examples(file):
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
           X.append(np.array(X_n).T)
           X_n = []
       elif line != '\n':
           attributes = line.strip().split(",")
           assert len(attributes) == 136
           X_n_row = [np.float64(attribute) for attribute in attributes[15:135]]
           X_n.append(X_n_row)
    
    examples = np.array(X)
    return examples

def split_subject_examples(subject_examples, time_steps, normalized=True):
    section_num = subject_examples[0].shape[0] / time_steps # 135/time_steps
    input_size = subject_examples[0].shape[1] 
    
    X = np.array([]).reshape([0, time_steps, input_size])
    for subject_example in subject_examples:
        sub_subject_examples = []
        for sub_subject_example in np.split(subject_example, section_num):
            if normalized:
               sub_subject_example = normalize(sub_subject_example, time_steps)
            sub_subject_examples.append(sub_subject_example)
        
        X = np.concatenate((X, sub_subject_examples))
    
    return X

def split_subject_labels(subject_labels, time_steps):
    labels = []
    expanded_coeff = 120/time_steps
    for line in subject_labels:
        expanded_line_labels = [line] * expanded_coeff
        labels = np.concatenate((labels, expanded_line_labels))
    
    return labels

def get_data_set(examples, labels, start, end, time_steps, shuffle=False):
    dataset_examples = examples[start:end]
    dataset_labels = labels[start:end]
    dataset_examples = split_subject_examples(dataset_examples, time_steps)
    dataset_labels = split_subject_labels(dataset_labels, time_steps)
    assert dataset_examples.shape[0] == dataset_labels.shape[0]
    if shuffle:
        num_dataset_examples = dataset_examples.shape[0]
        perm = np.arange(num_dataset_examples)
        np.random.shuffle(perm)
        dataset_examples = dataset_examples[perm]
        dataset_labels = dataset_labels[perm]
        
    dataset = Dataset(dataset_examples, dataset_labels)
    return dataset

def read_data_sets(data_dir, time_steps, shuffle=True):
    
    with open(data_dir + '/encoded_samples.txt', 'r') as f:
        subject_examples = extract_subject_examples(f)
    with open(data_dir + '/encoded_labels.txt', 'r') as f:
        subject_labels = []
        for line in f:
            subject_labels.append(line.strip())
        subject_labels = np.array(subject_labels)
    
    assert subject_examples.shape[0] == subject_labels.shape[0]
    
    num_subjects = subject_examples.shape[0]
    if shuffle:
        perm = np.arange(num_subjects)
        np.random.shuffle(perm)
        subject_examples = subject_examples[perm]
        subject_labels = subject_labels[perm]
    
    train = get_data_set(subject_examples, 
                         subject_labels, 
                         0, 
                         int(0.7*num_subjects), 
                         time_steps, 
                         shuffle)
    test =  get_data_set(subject_examples, 
                         subject_labels, 
                         int(0.7*num_subjects), 
                         num_subjects, 
                         time_steps)
    validation = get_data_set(subject_examples, 
                              subject_labels, 
                              0, 
                              int(0.7*num_subjects), 
                              time_steps)
    
    return Datasets(train=train, test=test, validation=validation)