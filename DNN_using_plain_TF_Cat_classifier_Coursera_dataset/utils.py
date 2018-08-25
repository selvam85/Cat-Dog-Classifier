import numpy as np
import h5py
import math

def load_dataset_cat_vs_non_cat():
    
    # h5py.File() returns a File object. 
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', mode = 'r')
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', mode = 'r')
    #print(type(train_dataset)) # Prints <class 'h5py._hl.files.File'>
    
    #Check the contents of the File object. h5py File acts like a dictionary. So, we can list the keys it contains
    #print(list(train_dataset.keys()))
    #print(list(test_dataset.keys()))
    
    train_set_x_orig = np.array(train_dataset['train_set_x'][:])
    train_set_y_orig = np.array(train_dataset['train_set_y'][:])
    
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])
    
    #Since both train and test classes contain the same two classes just returning back one of them
    classes = np.array(test_dataset['list_classes'][:])
    
    #print('Train X shape', train_set_x_orig.shape)
    #print('Train Y shape', train_set_y_orig.shape)
    #print('Test X shape', test_set_x_orig.shape)
    #print('Test Y shape', test_set_y_orig.shape)
    
    train_set_y_orig = train_set_y_orig.reshape(train_set_y_orig.shape[0], 1)
    test_set_y_orig = test_set_y_orig.reshape(test_set_y_orig.shape[0], 1)
    #print('Train Y shape', train_set_y_orig.shape)
    #print('Test Y shape', test_set_y_orig.shape)
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def create_random_mini_batches(X, Y, mini_batch_size = 32):
    
    n_samples = X.shape[0]                  
    mini_batches = []
    
    p = list(np.random.permutation(int(n_samples)))
    X_shuffled = X[p, :]
    Y_shuffled = Y[p, :]
    
    n_mini_batches = math.floor(n_samples / mini_batch_size)
    for i in range(n_mini_batches):
        
        start_pos = i * mini_batch_size
        end_pos = start_pos + mini_batch_size
        
        X_mini_batch = X_shuffled[start_pos : end_pos, :]
        Y_mini_batch = Y_shuffled[start_pos : end_pos, :]
         
        mini_batch = (X_mini_batch, Y_mini_batch)
        mini_batches.append(mini_batch)
    
    if n_samples % mini_batch_size != 0:
        
        start_pos = n_mini_batches * mini_batch_size
        
        X_mini_batch = X_shuffled[start_pos : n_samples, :]
        Y_mini_batch = Y_shuffled[start_pos : n_samples, :]
        
        mini_batch = (X_mini_batch, Y_mini_batch)
        mini_batches.append(mini_batch)
    
    return mini_batches
