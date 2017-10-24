# TensorFlow Neural Network Lab
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[<img src="http://yaroslavvb.com/upload/notMNIST/nmn.png" alt="notMNIST dataset samples" />](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html)

It turns out an easy exercise. My accuracy is 81%. 

<img src="image\Problem3.JPG" width="480" alt="Accuracy" />

At first, the PIL cannot work with Python 3.6.3. So I use the ‘imread’ from ‘matplotlib.pyplot’. Then, the notMNIST data set contains many error figures. Many figures that are all black, which means every pixel equals 255. Obviously, they are useless for classifier. My function ‘deleteNan’ will remove these figures. At last, ‘normalize_grayscale’ can handle both vectors and matrices. I have not turned some parameters before I got an accuracy over 81%. 
