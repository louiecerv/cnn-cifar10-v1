#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


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
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    
    st.subheader('Display Samples from the Dataset')
    with st.echo(code_location='below'):
        if st.button('Display Sample Data'):
            # Define class names
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

            # Print the first 20 images
            fig = plt.figure(figsize=(6,6))
            for i in range(20):
                plt.subplot(5, 4, i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(train_images[i], cmap=plt.cm.binary)
                plt.xlabel(class_names[train_labels[i][0]])
            st.pyplot(fig)
            
        if st.button('Run the Neural Network'):
            #set the number of hidden layers
            neurons = st.slider('No. of neurons in the hidden layer', 5, 15, 10)
            #set the number or iterations
            epochs = st.slider('Number of epochs', 50, 250, 100, 10)
            
    st.write('In this version of the MLP we used the Keras library running on Tensorflow.  \
            Keras is a high-level neural network library written in Python that can run \
            on top of TensorFlow, Theano, and other machine learning frameworks. \
            It was developed to make deep learning more accessible and easy to use \
            for researchers and developers.  TensorFlow provides a platform for \
            building and deploying machine learning models. It is designed to \
            be scalable and can be used to build models ranging from small experiments\
            to large-scale production systems. TensorFlow supports a wide range of \
            machine learning algorithms, including deep learning, linear regression, \
            logistic regression, decision trees, and many others.')        
   
#run the app
if __name__ == "__main__":
    app()
