import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.metrics import confusion_matrix

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator #array_to_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # Dropout

hyper_epochs = 100

def create_datagen(directory, img_size, batch_size, augment=False):
    """
    Create a data generator for the given directory.

    param directory: Path to the image directory.
    param img_size: Tuple of (height, width) for resizing the images.
    param batch_size: Number of images to process at once.
    param augment: Boolean, whether to apply data augmentation.
    return: A data generator for the specified directory.

    :ImageDataGenerator is an essential tool for image processing in deep learning. It simplifies data augmentation, makes memory usage more efficient, and integrates well with the model training pipeline in Keras, all of which are crucial for developing robust image-based machine learning models



    """
    if augment:
        # Apply data augmentation for training data
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
    else:
        # No data augmentation for validation/test data
        datagen = ImageDataGenerator(rescale=1./255)

    generator = datagen.flow_from_directory(
        directory,
        target_size=img_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='binary'  # or 'categorical', depending on your label configuration
    )

    return generator

# Paths to your datasets
train_dir = 'C:/Users/shiva/Documents/Data_Science_Projects/Pneumonia X-Ray_Tensorflow/train'
val_dir = 'C:/Users/shiva/Documents/Data_Science_Projects/Pneumonia X-Ray_Tensorflow/val'
test_dir = 'C:/Users/shiva/Documents/Data_Science_Projects/Pneumonia X-Ray_Tensorflow/test'

# Image parameters
img_height, img_width = 224, 224
batch_size = 32

# Create data generators
train_gen = create_datagen(train_dir, (img_height, img_width), batch_size, augment=True)
val_gen = create_datagen(val_dir, (img_height, img_width), batch_size, augment=True)
test_gen = create_datagen(test_dir, (img_height, img_width), batch_size)

# Image Visualisation
'''
def display_images(generator, num_images=5):
    """
    Display the first few images from a data generator.

    :param generator: The data generator created by ImageDataGenerator.
    :param num_images: Number of images to display.
    """
    data_batch, labels_batch = next(generator)

    plt.figure(figsize=(10, num_images * 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        # Reshape and display image
        image = data_batch[i]
        if image.shape[2] == 1:  # Grayscale image
            plt.imshow(image[:, :, 0], cmap='gray')
        else:  # RGB image
            plt.imshow(image)
        plt.title(f'Label: {labels_batch[i]}')
        plt.axis('off')
    plt.show()

# Call the function with your generator
display_images(train_gen)  
'''
model = Sequential([
    # Convolutional layer with 32 filters, kernel size of 3x3, and ReLU activation
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width,1 )),
    # Max pooling layer with a 2x2 window
    MaxPooling2D((2, 2)),
    # Another convolutional layer with 64 filters
    Conv2D(64, (3, 3), activation='relu'),
    # Another max pooling layer
    MaxPooling2D((2, 2)),
    # Flatten layer to convert 3D data to 1D
    Flatten(),
    # Dense layer with 64 neurons
    Dense(64, activation='relu'),
    # Output layer with a number of neurons equal to the number of classes, using softmax for multi-class classification
    Dense(hyper_epochs, activation='softmax') 
])


'''
Dense(num_classes=2, activation='softmax') in a CNN model built with Keras serves a specific purpose, particularly in the context of classification tasks:

Dense: This refers to a fully connected neural network layer. Each neuron in this layer receives input from all neurons of the previous layer, making it "dense".

num_classes: This is the number of neurons in the Dense layer and should match the number of classes in your classification task. For example, if you're classifying images into 10 different categories (like in CIFAR-10), num_classes would be 10.

activation='softmax': The softmax activation function is used in multi-class classification problems. It converts the output of the last dense layer into a probability distribution, so that each neuron's output (corresponding to each class) represents the probability that the input belongs to a particular class

'''


# Compiling of model
#This step involves specifying the optimizer, loss function, and metrics for training.

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model


model.fit(train_gen, epochs=10, validation_data=(val_gen))

# Evaluate the model 
test_loss, test_accuracy = model.evaluate(test_gen)
print('Test accuracy:', test_accuracy)

# Making predictions

predictions = model.predict_generator(test_gen)
predictions[predictions <= 0.5] = 0
predictions[predictions > 0.5] = 1

'''
The code then applies a threshold to these predictions to convert them from continuous values (probabilities) into binary categories (0 or 1, representing the two classes).

predictions[predictions <= 0.5] = 0: This line sets any prediction that is less than or equal to 0.5 to 0. In a binary classification context, this typically represents the negative class.

predictions[predictions > 0.5] = 1: This line sets any prediction greater than 0.5 to 1, representing the positive class.

In many binary classification problems, especially those involving probabilities (like in logistic regression or neural networks), a threshold of 0.5 is a common default. It means:

If the model predicts a probability greater than 0.5, the outcome is classified as 1 (or 'positive').
If the probability is less than or equal to 0.5, it's classified as 0 (or 'negative').
This threshold is especially important in medical or sensitive contexts, as different thresholds might be set based on the relative importance of false positives vs. false negatives.

'''


#Confusion Matrix

cf = pd.DataFrame(data=confusion_matrix(test_gen.classes,predictions , labels=[0, 1]),
                  index=["Actual Normal", "Actual Pneumonia"],
                  columns=["Predicted Normal", "Predicted Pneumonia"])
cf