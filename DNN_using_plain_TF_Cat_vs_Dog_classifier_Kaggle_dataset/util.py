import numpy as np
import h5py
import math
from random import shuffle

def load_dataset(filepath):
    
    with h5py.File(filepath, 'r') as f:
        
        X_train = np.array(f['train_data'][:])
        Y_train = np.array(f['train_labels'][:])
        X_cval = np.array(f['val_data'][:])
        Y_cval = np.array(f['val_labels'][:])
        
        Y_train = Y_train.reshape(Y_train.shape[0], 1)
        Y_cval = Y_cval.reshape(Y_cval.shape[0], 1)
        
    return X_train, Y_train, X_cval, Y_cval

def normalize_data(X, batch_size = 5000):
    n_samples = X.shape[0]
    n_batches = int(math.ceil(n_samples / batch_size))
    
    X = X.astype(float)
    
    for i in range(n_batches):
        
        start_index = i * batch_size
        end_index = min(start_index + batch_size, n_samples)
        
        batch = X[start_index : end_index, :]
        #print('Batch Shape =', batch.shape)
        batch = batch / 255
        
        X[start_index : end_index, :] = batch
        
        print('{} images normalized'.format((i + 1) * (end_index - start_index)))
    
    return X

def create_random_mini_batches(X, Y, n_classes = 2, mini_batch_size = 32, one_hot_vector_flag = True):
    
    n_samples = X.shape[0]
    n_mini_batches = int(math.ceil(n_samples / mini_batch_size))
    mini_batches_indices = list(range(n_mini_batches))
    
    shuffle(mini_batches_indices)
    
    mini_batches = []
    
    for n, i in enumerate(mini_batches_indices):
        
        start_index = i * mini_batch_size
        end_index = min(start_index + mini_batch_size, n_samples)
        curr_mini_batch_size = end_index - start_index # end index - start index = mini_batch_size
        
        mini_batch_x = X[start_index : end_index, :]
    
        if one_hot_vector_flag:
            mini_batch_y_one_hot = np.zeros((curr_mini_batch_size, n_classes)) 
            mini_batch_y_one_hot[np.arange(curr_mini_batch_size), np.squeeze(mini_batch_y)] = 1
            mini_batches.append((mini_batch_x, mini_batch_y_one_hot))
        else:
            mini_batch_y = Y[start_index : end_index, :]
            mini_batches.append((mini_batch_x, mini_batch_y))
    
    return mini_batches