# Detecting Skin Infections using CNNs and Transfer Learning 

Aim : To be able to identify the 

## Exploratory Data Analysis

## Model A : Replication of the AlexNet model using Sequential API 
![Figure 1: Diagram of AlexNet model](image.png)

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


## Model A : Results

The code for Model A can be found in the model_A.ipynb notebooks located in the notebooks folder of this repository, functions have been fully annotated using doc strings thus enabling ease of understanding of the models contained within functions.

![Figure 2 : Learning curves to asses change of loss, accuracy and learning rate ](image-1.png)


| Metric             | Value                      |
|--------------------|----------------------------|
| Train Accuracy      | 0.7316                     |
| Test Accuracy       | 0.7639                     |
| Train Loss          | 0.7845                     |
| Test Loss           | 1.0089                     |

As can be seen from the model metrics, the train and test accuracies are relatively low, and the gap between train and test loss is quite significant. This suggests that the model is **underfitting**, meaning it is not able to learn patterns effectively from the data provided. This underfitting could be caused by excessive regularization techniques, such as Dropout and BatchNormalization, which may be too aggressively constraining the model's learning capacity.

## Experiment 2 : Removing the dropout layers from AlexNet model

![Figure 3 : Learning curves to asses change of loss, accuracy and learning rate without dropout regularization](image-2.png)

| Metric            | Value                      |
|-------------------|----------------------------|
| Train Accuracy     | 0.9242                     |
| Test Accuracy      | 0.8841                     |
| Train Loss         | 0.2163                     |
| Test Loss          | 0.6179                     |

Removing the dropout regularization layers, we find that the model drastically improves in being able to predict the class labels with a 3% discrepancy between test and training accuracy. The train loss and test loss indicate that while the model is fitting the training data well, there is still some overfitting occurring, as evidenced by the higher test loss compared to the train loss. This suggests that the model generalizes slightly less effectively on unseen data, despite the improved accuracy.

## Model A Conclusions 

The **AlexNet architecture** was implemented using the **TensorFlow/Keras** Python package. Two experiments were conducted, and their outcomes are summarized below:

- **Experiment 1**: AlexNet with Batch Normalization applied to the convolutional layers and Dropout applied to the fully connected layers.
- **Experiment 2**: AlexNet without Dropout applied to the fully connected layers.

## Results of Experiment 1:
In Experiment 1, we observed **underfitting** of the model, which was indicated by a noticeable difference between the test and training scores. This underfitting was attributed to two factors:

1. **No Pre-Trained Weights**:  
   Since the model was trained from scratch without pre-trained weights, it may not have had sufficient exposure to data to effectively learn complex patterns. Leveraging a pre-trained model could improve performance by providing a better starting point.

2. **Over-Regularization**:  
   The second hypothesis was that the model might have been **over-regularized** due to the combined use of Batch Normalization and Dropout. To test this, we proceeded to **Experiment 2**, where the Dropout layers were removed to evaluate their impact on the model's performance.

## Results of Experiment 2:
In Experiment 2, the removal of the Dropout layers resulted in a **marked improvement in accuracy**. However, the model still exhibited slight **overfitting**, as indicated by the discrepancy in the loss metrics between the training and test scores. This suggests that while the model's performance improved, it still struggled to generalize fully to unseen data.

## Conclusion:
Based on these findings, it was concluded that while removing Dropout improved accuracy, it was not sufficient to eliminate overfitting. To further optimize performance, a **pre-trained model** will be utilized as a benchmark to compare against these "fresh" models, potentially leading to better results in terms of both accuracy and generalization.
