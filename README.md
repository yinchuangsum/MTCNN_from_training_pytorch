# MTCNN_from_training_pytorch

MTCNN is a Convolution Neural Network used in face detection.\
It is based on the paper [Zhang, K et al. (2016)](https://arxiv.org/abs/1604.02878)\
![alt text](https://i.ibb.co/WcZ7Rvc/test-img.jpg) ![alt text](https://i.ibb.co/6P225L3/test-img.jpg)
---
## Training
train MTCNN model using [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset
1. Use generate_data.py to generate & resize data into suitable image size for training. (modify the file save path)
2. Run train_pnet.py, train_rnet.py and train_onet.py to train the 3 network.
---

## Detection
1. Put images that needed to be detected in the test_images folder.
2. Run detect.py.
3. Output images will be in the output folder.


