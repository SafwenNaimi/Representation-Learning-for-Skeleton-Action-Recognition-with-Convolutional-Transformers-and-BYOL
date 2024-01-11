# import package
import mpose
import numpy as np

# initialize and download data
dataset = mpose.MPOSE(pose_extractor='openpose', 
                      split=1)
                      #preprocess='scale_and_center') 
                      #data_dir='./data/')

# print data info 
dataset.get_info()
print(dataset)

# get data samples (as numpy arrays)
X_train, y_train, X_test, y_test = dataset.get_data()
print(X_train.shape)
#for y in y_train: 
    #print(y)

#print(X_train[0][0])