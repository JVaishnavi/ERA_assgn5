# Assignment 5 ERA course

## MNIST dataset
The MNIST database (Modified National Institute of Standards and Technology database[1]) is a large database of handwritten digits that is commonly used for training various image processing systems.

This repo contains code to identify handwritten numbers.

Input: Image of size 28x28
Output: Class value between 0-9 each value signifying which digit the image shows

## Folder structure

- `model.py`: contains the model (Convolutional neural network model)
- `utils.py`: contains utility functions for data training, and testing
- `S5.ipynb`: notebook to run the model on the dataset

## Model Architecture

### Layer 1:

- Network: CNN
- Kernel: 3x3
- Padding: 0
- Input channel size: 1
- Output channel size: 32
- Input dimension: 28x28x1
- Output dimension: 26x26x32
- Activation function: Relu

### Layer 2:

- Network: CNN
- Kernel: 3x3
- Padding: 0
- Input channel size: 32
- Output channel size: 64
- Input dimension: 26x26x32
- Output dimension: 24x24x64

### Layer 3:

- Network: Max Pooling
- Input channel size: 64
- Output channel size: 64
- Input dimension: 24x24x64
- Output dimension: 12x12x64
- Activation function: Relu

### Layer 4:

- Network: CNN
- Kernel: 3x3
- Padding: 0
- Input channel size: 64
- Output channel size: 128
- Input dimension: 12x12x64
- Output dimension: 10x10x128
- Activation function: Relu

### Layer 5:

- Network: CNN
- Kernel: 3x3
- Padding: 0
- Input channel size: 128
- Output channel size: 256
- Input dimension: 10x10x128
- Output dimension: 8x8x256

### Layer 6:

- Network: Max Pooling
- Input channel size: 256
- Output channel size: 256
- Input dimension: 8x8x256
- Output dimension: 4x4x256
- Activation function: Relu

### Layer 7:

- Network: FC
- Input dimension: 8x8x256 (i.e. 4096 values)
- Output dimension: 50
- Activation function: Relu

### Layer 8:

- Network: FC
- Input dimension: 50
- Output dimension: 10 (Final output dimension)
- Activation function: Log softmax
