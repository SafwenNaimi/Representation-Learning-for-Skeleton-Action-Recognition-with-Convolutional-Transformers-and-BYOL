import tensorflow as tf
import numpy as np
from mpose import MPOSE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#chnage the labels according to the dataset
labels = { # 20 Classes
    "check-watch":0,
    "cross-arms":1,
    "get-up":2,
    "kick":3,
    "pick":4,
    "point":5,
    "punch":6, 
    "scratch-head":7, 
    "sit-down":8,
    "turn-around":9,
    "walk":10,
    "wave":11}


def load_kinetics(config, fold=0):
    
    X_train = np.load('/media/Datasets/kinetics/train_data_joint.npy') #[...,0] # get first pose
    X_train = np.moveaxis(X_train,1,3)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
    X_train = X_train[:,::config['SUBSAMPLE'],:]
    y_train = np.load('/media/Datasets/kinetics/train_label.pkl', allow_pickle=True)
    y_train = np.transpose(np.array(y_train))[:,1].astype('float32') # get class only
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=config['VAL_SIZE'],
                                                      random_state=config['SEEDS'][fold],
                                                      stratify=y_train)
    
    X_test = np.load('/media/Datasets/kinetics/val_data_joint.npy') #[...,0] # get first pose
    X_test = np.moveaxis(X_test,1,3)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
    X_test = X_test[:,::config['SUBSAMPLE'],:]
    y_test = np.load('/media/Datasets/kinetics/val_label.pkl', allow_pickle=True)
    y_test = np.transpose(np.array(y_test))[:,1].astype('float32') # get class only
    
    train_gen = callable_gen(kinetics_generator(X_train, y_train, config['BATCH_SIZE']))
    val_gen = callable_gen(kinetics_generator(X_val, y_val, config['BATCH_SIZE']))
    test_gen = callable_gen(kinetics_generator(X_test, y_test, config['BATCH_SIZE']))
    
    return train_gen, val_gen, test_gen, len(y_train), len(y_test)


def load_data(dataset, split, verbose=False, legacy=False):
    
    if legacy:
        return load_dataset_legacy(data_folder='E:/AcT/')
    
    d = MPOSE(pose_extractor=dataset, 
                    split=split, 
                    preprocess=None, 
                    velocities=True, 
                    remove_zip=False)
    
    if 'legacy' not in dataset:
        d.reduce_keypoints()
        d.scale_and_center()
        d.remove_confidence()
        d.flatten_features()
        #d.reduce_labels()
        return d.get_data()
    
    elif 'vitpose' in dataset:
        X_train, y_train, X_test, y_test = d.get_data()
        print(X_train.shape)
        return X_train, transform_labels(y_train), X_test, transform_labels(y_test)
        #return X_train, (y_train), X_test, (y_test)
    else:
        return d.get_data()
        

def random_flip(x, y):
    time_steps = x.shape[0]
    #print(time_steps)
    n_features = x.shape[1]
    if not n_features % 2:
        x = tf.reshape(x, (time_steps, n_features//2, 2))

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        if choice >= 0.5:
            x = tf.math.multiply(x, [-1.0,1.0])
    else:
        x = tf.reshape(x, (time_steps, n_features//3, 3))

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        if choice >= 0.5:
            x = tf.math.multiply(x, [-1.0,1.0,1.0])
    x = tf.reshape(x, (time_steps,-1))
    return x, y


def random_noise(x, y):
    time_steps = tf.shape(x)[0]
    n_features = tf.shape(x)[1]
    noise = tf.random.normal((time_steps, n_features), mean=0.0, stddev=0.03, dtype=tf.float64)
    x = x + noise
    return x, y


def one_hot(x, y, n_classes):
    #y = int(y)
    return x, tf.one_hot(y, n_classes)


def kinetics_generator(X, y, batch_size):
    while True:
        ind_list = [i for i in range(X.shape[0])]
        shuffle(ind_list)
        X  = X[ind_list,...]
        y = y[ind_list]
        for count in range(len(y)):
            yield (X[count], y[count])
        
        
def callable_gen(_gen):
        def gen():
            for x,y in _gen:
                 yield x,y
        return gen
    
    
def transform_labels(y):
    y_new = []
    for i in y:
        y_new.append(labels[i])
    return np.array(y_new)


def load_dataset_legacy(data_folder, verbose=True):
    X_train = np.load(data_folder + 'X_train.npy')
    y_train = np.load(data_folder + 'Y_train.npy', allow_pickle=True)
    y_train = transform_labels(y_train)
    
    X_test = np.load(data_folder + 'X_test.npy')
    y_test = np.load(data_folder + 'Y_test.npy', allow_pickle=True)
    y_test = transform_labels(y_test)
    
    if verbose:
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")

        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
    return X_train, y_train, X_test, y_test
