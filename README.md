# Five_Flower_Classification
A project on classification of five flowers, namely daisy, dandelion, rose, sunflower and tulip.
Entire project is in Python and the I have used Tensorflow Keras for model creation.
Dataset Link : https://www.kaggle.com/alxmamaev/flowers-recognition

1) First I tried implementing a CNN model but only managed to reach around 64% accuracy in 50 epochs. 

![cnn_5_layer](https://user-images.githubusercontent.com/59522832/138867040-b3c2e7ba-05a0-4611-b209-13b814717d1b.jpg)
CNN has 5 layers of Convolution and MaxPooling stacks. Initially, CNN started with 24% but managed to get 64% accuracy at the end of 50 epochs.

2) Secondly, I tried InceptionV3 model which increased my accuracy to 85%.

![Inception_V3_plot](https://user-images.githubusercontent.com/59522832/138868304-7630f1a3-ca55-478b-a710-d65c9d700887.jpg)

3) Thirdly, I experimented with Resnet152V2 model and it also gave me around 85% accuracy.

![Resnet 152V2_plot](https://user-images.githubusercontent.com/59522832/138868382-b8d0339c-7ba4-42f0-99fa-3b663b110a32.jpg)

Transfer Learning helped me to increase accuracy on the flower dataset by almost 20%.
