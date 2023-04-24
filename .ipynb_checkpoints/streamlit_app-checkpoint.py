#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.datasets import cifar10

# Define a custom callback function to update the Streamlit interface
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get the current loss and accuracy metrics
        loss = logs['loss']
        accuracy = logs['accuracy']
        
        # Update the Streamlit interface with the current epoch's output
        st.write(f"Epoch {epoch}: loss = {loss:.4f}, accuracy = {accuracy:.4f}")

        
train_images = []
train_labels = []
test_images = []
test_labels = [] 
    
# Define the Streamlit app
def app():

    
    st.title('Deep Learning Using Convolutional Neural Network on Tensorflow and Keras')
    st.subheader('by Louie F. Cervantes M.Eng., WVSU College of ICT')
    st.subheader('The Convolutional Neural Network')

    st.write('A convolutional neural network (CNN) is a type of artificial neural network \
    that is commonly used for image recognition and computer vision tasks. The key idea \
    behind a CNN is to use filters, or kernels, that can be convolved with the input data \
    to extract relevant features.')
    
    st.write('A CNN typically consists of multiple layers, including convolutional layers, \
    pooling layers, and fully connected layers. The convolutional layers are responsible \
    for applying the filters to the input data, which helps to identify features such as edges, \
    corners, and textures. The pooling layers are used to downsample the output of the \
    convolutional layers, reducing the spatial dimensions of the data and helping to make \
    the network more efficient. The fully connected layers are used to perform the final \
    classification or regression task, taking the output of the previous layers and \
    producing a final output.')

    st.write('During the training process, the weights of the filters and the parameters \
    of the fully connected layers are adjusted through backpropagation, using a loss function \
    such as cross-entropy or mean squared error. This allows the network to learn to \
    recognize patterns and features in the input data, and to make accurate predictions on new data.')
 
    st.subheader('The CIFAR10-Dataset')
    st.write('CIFAR-10 is a popular image classification dataset that consists of 60,000 \
    32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided \
    into 50,000 training images and 10,000 testing images.')
    st.write('The 10 classes in the CIFAR-10 dataset are: \
    \n1. Airplane \
    \n2. Automobile \
     \n3. Bird \
     \n4. Cat \
     \n5. Deer \
     \n6. Dog \
     \n7. Frog \
     \n8. Horse \
     \n9. Ship \
     \n10. Truck')
    
    st.write('The images in the CIFAR-10 dataset are low-resolution (32x32) and have a color \
    depth of 3 (RGB). The dataset was collected by researchers at the Canadian Institute for \
    Advanced Research (CIFAR) and is commonly used as a benchmark for image classification tasks.')
    
    # Load the CIFAR-10 dataset
    global train_images, train_labels, test_images, test_labels
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    
    st.subheader('Display Samples from the Dataset')
    with st.echo(code_location='below'):
        if st.button('Display Sample Data'):
            # Define class names
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

            # Print the first 20 images
            fig = plt.figure(figsize=(6,8))
            for i in range(20):
                plt.subplot(5, 4, i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(train_images[i], cmap=plt.cm.binary)
                plt.xlabel(class_names[train_labels[i][0]])
            st.pyplot(fig)
            
        #set the number or iterations
        epochs = st.slider('Number of epochs', 3, 5, 3, 1) 
        if st.button('Run the Neural Network'):
            
            # Normalize pixel values between 0 and 1
            train_images = train_images.astype("float32") / 255.0
            test_images = test_images.astype("float32") / 255.0

            # Convert class labels to one-hot encoded vectors
            num_classes = 10
            train_labels = keras.utils.to_categorical(train_labels, num_classes)
            test_labels = keras.utils.to_categorical(test_labels, num_classes)

            # Define the CNN architecture
            model = keras.Sequential(
                [
                    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Conv2D(64, (3, 3), activation="relu"),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Conv2D(128, (3, 3), activation="relu"),
                    layers.Flatten(),
                    layers.Dense(128, activation="relu"),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )

            # Compile the model
            model.compile(
                loss="categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"],
            )

            # Train the model
            batch_size = 64
            model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_images, test_labels), callbacks=[CustomCallback()])

            # Evaluate the model on the test set
            score = model.evaluate(test_images, test_labels, verbose=0)
            print("Test loss:", score[0])
            print("Test accuracy:", score[1])

    st.write('Conclusion')        
   
#run the app
if __name__ == "__main__":
    app()
