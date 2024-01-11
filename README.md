# Representation-Learning-for-Skeleton-Action-Recognition-with-Convolutional-Transformers-and-BYOL


![new_architecture drawio_ drawio (1)](https://github.com/SafwenNaimi/Representation-Learning-for-Skeleton-Action-Recognition-with-Convolutional-Transformers-and-BYOL/assets/55064537/9c6fe0fd-cddf-45cb-b941-801a5146fa37)

To extract robust and more generalizable skeleton action recognition features, large amounts of well-curated data are typically required, which is a challenging task hindered by annotation costs. Therefore, unsupervised representation learning is of prime importance to leverage unlabeled skeleton data. In this work, we investigate unsupervised representation learning for skeleton action recognition. 

This repository complements our paper, providing a reference implementation of the method as described in the paper. Please contact the authors for enquiries regarding the code.

# Dependencies
We tested our code on the following environment.

* tensorflow 2.6.0
* Python 3.8.5
* numpy 1.23.5

Install python libraries with:

    pip install -r requirements.txt

# Data preparation for NW-UCLA:
1. Download raw data from http://users.eecs.northwestern.edu/~jwa368/data/multiview_action_videos.tgz
2. Place the videos in data/NW-UCLA in that form:

        ├── data
          ├── NW-UCLA
             ├── carry
                ├──  vid_1
                ├──  vid_2
                ........
             ├── doffing
                ├──  vid_1
                ├──  vid_2
                ........
             ├── donning
                ├──  vid_1
                ├──  vid_2
                ........
             ├── ....
             ├── ....
   
4. Download ViTPose Huge model: https://huggingface.co/JunkyByte/easy_ViTPose/blob/main/torch/coco_25/vitpose-h-coco_25.pth
5. Place the ViTPose Huge model in data/
6. Install the requirements for the data preparation:
   
       cd data/       
       pip install -r requirements.txt
7. Generate keypoints for each video:
    
       python data/keypoints_finder.py
8. Generate npy file for each class:

       python data/creating_npy.py     

# BYOL Pretraining:
To run a byol training process:

       python BYOL_pretraining/byol_train_script.py      
