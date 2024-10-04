# Detecting Skin Infections using CNNs and Transfer Learning 

Aim : To be able to identify the 

## Exploratory Data Analysis

## Model A : Replication of the AlexNet model using Sequential API 
![alt text](image.png)

This implementation of AlexNet consists of **8 convolutional layers**, followed by a **feed-forward neural network**. We're using blank networks with **no pre-trained weights** to gain hands-on experience in building a deep neural network from scratch using the Tensorflow and Keras packages.

## Key Concepts:

1. **Convolutional Layers**:  
   These layers extract features from an image.  
   After each convolution operation (between image and filter/kernel), we apply **Batch Normalization**.

2. **Batch Normalization**:  
   This process normalizes the inputs by scaling them using their variance and mean.  
   It also applies an offset using the **scale** and **shift** parameters to improve network performance.

3. **Pooling Layers**:  
   Pooling layers **aggregate feature maps** by shrinking the image size.  
   This reduces computational complexity while retaining essential features.

4. **Feed-Forward Neural Network**:  
   After the convolutional layers, we have a fully connected feed-forward network.  
   To prevent overfitting, **Dropout layers** are used to regularize the predictions.

5. **Output Layer**:  
   The output layer is a **Softmax function** used for multiclass classification.  
   In this case, we are classifying multiple types of **skin infections**.

---

By building this network, we focus on understanding the inner workings of a deep neural network architecture, including how each layer contributes to the final prediction.