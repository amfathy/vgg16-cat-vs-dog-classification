# VGG16 Cat vs Dog Image Classification

## Description
This project uses the VGG16 pre-trained model to classify images of cats and dogs. The model leverages transfer learning by freezing the layers of the VGG16 model and adding custom layers for classification. The project also includes data preprocessing using `ImageDataGenerator` for training and testing.

## Features
- **Transfer Learning with VGG16**: Utilizes the pre-trained VGG16 model with custom classification layers on top.
- **Image Augmentation**: Applies data augmentation techniques like shear, zoom, and horizontal flip to training images.
- **Evaluation**: The model is evaluated on a test set to measure accuracy.

## Requirements
- Python 3.x
- TensorFlow 2.x
- NumPy
- Keras

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/amfathy/vgg16-cat-vs-dog-classification.git
