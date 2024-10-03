import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Flatten

def create_alexnet(num_classes : int, loss_function="categorical_crossentropy", metrics : list = ['accuracy']) -> tf.keras.Model :
    
    """
    Creates a Convolutional Neural Network (CNN) model based on the AlexNet architecture, 
    with Batch Normalization layers for stabilizing the training. The model starts with an input
    image dimension of (227, 227, 3) and progressively reduces the spatial dimensions while 
    increasing the number of filters in each Conv2D/MaxPooling block.

    Args:
        num_classes (int): The number of output classes for the classification task.
        loss_function (str or callable): The loss function to use during model compilation. 
                                        Default is 'categorical_crossentropy'.
        metrics (list): A list of metrics to be evaluated by the model during training and testing. 
                        Default is ['accuracy'].
        
    Returns:
        model (tf.keras.Model): A compiled CNN model with batch normalization, ready for training.
    
    Example:
        model = create_alexnet(10, loss_function='categorical_crossentropy')
        model.summary()
    """
    model = Sequential([

        # 1st convolutional-pooling block
        Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation="relu", input_shape=(227,227,3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3,3), strides=(2,2)),

        # 2nd convolutional-pooling block
        Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), activation="relu", padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3,3), strides=(2,2)),

        # 3rd convolutional block
        Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation="relu", padding='same'),
        BatchNormalization(),
        Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation="relu", padding='same'),
        BatchNormalization(),
        Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation="relu", padding='same'),
        BatchNormalization(),

        # Flatten layer to convert 3D feature map into 1D Vector to feed into Neural Network
        Flatten(),

        # Fully connected (dense) layers block with dropout regularization
        Dense(4096, activation="relu"),
        Dropout(0.5),
        Dense(4096, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer='Adam', loss=loss_function, metrics=['accuracy'])

    return model
    
