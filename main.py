import os
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 
from PIL import Image

pneumonia_dir = 'C:\\Users\\miste\\OneDrive\\Desktop\\FYP\\IMG_xray\\train\\PNEUMONIA'
normal_dir = 'C:\\Users\\miste\\OneDrive\\Desktop\\FYP\\IMG_xray\\train\\NORMAL'
#IMG Array shape (num of images, height, width, depth)

#CNN Hyperparameters

IMG_SIZE = (256,256)        #Other img sizes for vaild padding with 3*3 conv: 320, 512    
FILTER_SHAPE = (3,3,1)      #(height, width, depth) 
NUM_FILTERS = 3
EPOCHS = 10
LR = 0.001                  #Learning rate

'''CNN'''

#Base Class for Activation Functions
class Activation:
    def __init__(self, activation, activation_der):
        self.activation = activation
        self.activation_der = activation_der

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_der(self.input))

#Sigmoid Activation Function
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_der(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_der)

#ReLU Activation Function
class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_der(x):
            return np.where(x > 0, 1, 0)

        super().__init__(relu, relu_der)

#Convolutional Layer
class Conv:

    def __init__(self, input_shape, filter_size, number_of_filters):
        input_height, input_width, input_depth  = input_shape # (height, width, depth)
        self.number_of_filters = number_of_filters
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (input_height - filter_size + 1, input_width - filter_size + 1, number_of_filters) # (height, width, num_filters/depth) 
        self.filters_shape = (number_of_filters, filter_size, filter_size, input_depth) # (num_filters, height, width, depth)
        self.biases = np.zeros((number_of_filters,1))

        #He Initialization for conv layer and Relu activation
        fan_in = filter_size * filter_size * input_depth
        sigma = sqrt(2 / fan_in)
        self.filters = np.random.normal(0, sigma, self.filters_shape)
         
    #Takes single image as input in the shape of (height, width, depth) and returns the output in the shape of (height, width, num_filters)
    def forward(self, input_data):
        self.input = input_data
        self.output = np.zeros(self.output_shape)
        for i in range(self.number_of_filters):
            filter = self.filters[i]
            for j in range(self.input_depth):
                self.output[:,:,i] += signal.correlate2d(self.input[:, :, j], filter[:, :, j], "valid")
            self.output[:,:,i] += self.biases[i] 
        return self.output

    def backward(self, output_gradient, learning_rate):  #Ooutput gradient shape (height, width, depth)
        filters_gradient = np.zeros(self.filters_shape) # (num_filters, height, width, depth)
        input_gradient = np.zeros(self.input_shape)     # (height, width, depth)

        for i in range(self.number_of_filters):
            filter = self.filters[i]
            for j in range(self.input_depth):
                #Update filters
                filters_gradient[i,:,:, j] = signal.correlate2d(self.input[:, :, j], output_gradient[:,:,i], "valid")
                #Update input gradient for next layer backprop
                input_gradient[:,:,j] += signal.convolve2d(output_gradient[:,:,i], filter[:, :, j], "full")

        self.filters -= learning_rate * filters_gradient 
        self.biases -= learning_rate * np.sum(output_gradient, axis=(0, 1)).reshape(-1, 1)
        return input_gradient
    
#Fully Connected Layer
class Dense:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.bias = np.zeros(self.output_size)

        #Initialize weights using Xavier Initialization
        sigma = sqrt(2/(self.input_size + self.output_size))
        self.weights = np.random.normal(0, sigma, (self.output_size, self.input_size))

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        output_gradient = output_gradient.reshape(self.bias.shape)
        self.bias -= learning_rate * output_gradient
        return input_gradient

#Max Pooling Layer
class MaxPool:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_data):
        self.input = input_data
        input_height, input_width, input_depth = input_data.shape
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1
        self.output = np.zeros((output_height, output_width, input_depth))
        

        for d in range(input_depth):
            for h in range(output_height):
                height_start = h * self.stride
                height_end = height_start + self.pool_size
                for w in range(output_width):
                    width_start = w * self.stride
                    width_end = width_start + self.pool_size

                    region = self.input[height_start:height_end, width_start:width_end, d]
                    self.output[h, w, d] = np.max(region)

        return self.output

    def backward(self,output_gradient):
        input_gradient = np.zeros(self.input.shape)
        input_height, input_width, input_depth = self.input.shape
        output_height, output_width, _ = output_gradient.shape

        for d in range(input_depth):
            for h in range(output_height):
                height_start = h * self.stride
                height_end = height_start + self.pool_size
                for w in range(output_width):
                    width_start = w * self.stride
                    width_end = width_start + self.pool_size

                    #Find the max value in maxpool region 
                    region = self.input[height_start:height_end, width_start:width_end, d]
                    max_val = np.max(region)

                    # Assign the gradient to the max value location in the input
                    for i in range(self.pool_size):
                        for j in range(self.pool_size):
                            if region[i, j] == max_val:
                                input_gradient[height_start + i, width_start + j, d] = output_gradient[h, w, d]

        return input_gradient

#Function to convert Convolutional output to Dense input
def Conv_to_Dense(input_data):
    input_shape = input_data.shape
    output_shape = (input_shape[0] * input_shape[1] * input_shape[2], 1)
    dense_input = np.reshape(input_data, output_shape)
    return dense_input, input_shape

#Function to convert Dense input to Convolutional input for backpropagation
def Dense_to_Conv(dense_gradient, input_shape):
    output_shape = (input_shape[0], input_shape[1], input_shape[2])
    return np.reshape(dense_gradient, output_shape)

#Binary Cross Entropy Loss Function
def bce(img_label, prediction):
    return -np.mean(img_label * np.log(prediction) + (1 - img_label) * np.log(1 - prediction))
  
#Derivative of Binary Cross Entropy Loss Function
def bce_der(img_label, prediction):
    return ((1 - img_label) / (1 - prediction) - img_label / prediction) / np.size(img_label)

#Function to train network
def train(epochs=EPOCHS,lr=LR):
    train_img, train_lbl = prepare_train_dataset(1,IMG_SIZE)    #depth, img_size

    conv = Conv(train_img[0].shape,3,NUM_FILTERS)
    relu = ReLU()
    pool = MaxPool()
    dense = Dense(127*127*3,1)
    sigmoid = Sigmoid()

    for epoch in range(epochs):
        error=0
        for img, lbl in zip(train_img, train_lbl):

            conv_output = conv.forward(img)

            relu_output = relu.forward(conv_output)
            
            pool_output = pool.forward(relu_output)
            
            dense_input, conv_shape = Conv_to_Dense(pool_output)
            
            dense_output = dense.forward(dense_input)
            
            sigmoid_output = sigmoid.forward(dense_output)
            
            error = bce(lbl, sigmoid_output)  #Binary Cross Entropy Loss to calculate error
            
            loss_gradient = bce_der(lbl, sigmoid_output)
            
            sigmoid_gradient = sigmoid.backward(loss_gradient)
            
            dense_gradient = dense.backward(sigmoid_gradient, LR)
            
            dense_To_conv_gradient = Dense_to_Conv(dense_gradient, conv_shape)
            
            pool_gradient = pool.backward(dense_To_conv_gradient)
            
            relu_gradient = relu.backward(pool_gradient)
            
            conv_gradient = conv.backward(relu_gradient, LR)

'''END OF CNN'''

# Function to preprocess the images : Grayscale -> Resize -> Normalize -> Return as NumPy array
def preprocess_data(directory, label, IMG_SIZE):
    images = []
    labels = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if filename.lower().endswith((".jpeg", ".jpg", ".png")):
            try:
                with Image.open(file_path).convert("L") as img:  # Convert to grayscale
                    img = img.resize(IMG_SIZE) 
                    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
                    images.append(img_array)
                    labels.append(label)  # Assign label: 1 for pneumonia, 0 for normal
            except Exception as e:
                print(f"Error processing file: {filename}")

        else:
             print(f"Skipping invalid file type: {filename}")

    return np.array(images), np.array(labels)

# Function to prepare the training dataset
def prepare_train_dataset(depth, img_size):
    pneumonia_img, pneumonia_label = preprocess_data(pneumonia_dir, 1, img_size)
    normal_img, normal_label = preprocess_data(normal_dir, 0, img_size)

    
    train_imgs = np.concatenate((pneumonia_img, normal_img), axis=0)
    train_lbls = np.concatenate((pneumonia_label, normal_label), axis=0)

    #TODO :Move suffle to shuffle dataset before each epoch
    indices = np.arange(train_imgs.shape[0])
    np.random.shuffle(indices)
    train_imgs = train_imgs[indices]
    train_lbls = train_lbls[indices]

    # Reshape for CNN input (number of Images, height, width, depth)
    train_imgs = train_imgs.reshape(train_imgs.shape[0], img_size[0], img_size[1], depth)

    print(f"Processed dataset shape: {train_imgs.shape}, Labels shape: {train_lbls.shape}")
    
    return train_imgs, train_lbls 

# Function to display the images
def display_img(train_imgs, train_lbls):
    for img, lbl in zip(train_imgs, train_lbls):
        print(f"Label: {lbl}")
        print(f"Image array:\n{img.shape}")

        # Convert the NumPy array back to an image
        img = (img * 255).astype(np.uint8)  # Denormalize the image
        img = img.squeeze()  # Remove the single channel dimension
        pil_img = Image.fromarray(img)
        
        # Display the image using PIL
        pil_img.show()
    
# EXECUTION

train()

'''
conv_output = conv.forward(train_img[0])
print(f"Conv Output Shape{conv_output.shape}")

relu_output = relu.forward(conv_output)
print(f"Relu Output Shape{relu_output.shape}")

pool_output = pool.forward(relu_output)
print(f"Pool Output Shape{pool_output.shape}")

dense_input, shape = Conv_to_Dense(pool_output)
print(f"Dense Input Shape{dense_input.shape}")

dense_output = dense.forward(dense_input)
print(f"Dense Output Shape{dense_output.shape}")
print(f"Dense Output: {dense_output}")

sigmoid_output = sigmoid.forward(dense_output)
print(f"Sigmoid Output{sigmoid_output}")

loss = bce(train_lbl[0], sigmoid_output)
print(f"Loss: {loss}")

#Backpropagation
loss_gradient = bce_der(train_lbl[0], sigmoid_output)
print(f"Loss Gradient: {loss_gradient}")

sigmoid_gradient = sigmoid.backward(loss_gradient, LR)
print(f"Sigmoid Gradient: {sigmoid_gradient}")

dense_gradient = dense.backward(sigmoid_gradient, LR)
print(f"Dense Gradient Shape: {dense_gradient.shape}")

dense_To_conv_gradient = Dense_to_Conv(dense_gradient, shape)
print(f"Dense to Conv Gradient Shape: {dense_To_conv_gradient.shape}")

pool_gradient = pool.backward(dense_To_conv_gradient)
print(f"Pool Gradient Shape: {pool_gradient.shape}")

relu_gradient = relu.backward(pool_gradient, LR)
print(f"Relu Gradient Shape: {relu_gradient.shape}")

conv_gradient = conv.backward(relu_gradient, LR)
print(f"Conv Gradient Shape: {conv_gradient.shape}")
'''
