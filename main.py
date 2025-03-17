import os
import datetime
import signal
import sys
import numpy as np
from math import sqrt
from scipy import signal 
from PIL import Image

#Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PNEUMONIA_TRAIN_DIR = os.path.join(BASE_DIR, 'IMG_xray', 'train', 'PNEUMONIA')  #IMG Array shape (num of images, height, width, depth)
NORMAL_TRAIN_DIR = os.path.join(BASE_DIR, 'IMG_xray', 'train', 'NORMAL')
PNEUMONIA_TEST_DIR = os.path.join(BASE_DIR, 'IMG_xray', 'test', 'PNEUMONIA')
NORMAL_TEST_DIR = os.path.join(BASE_DIR, 'IMG_xray', 'test', 'NORMAL')
WEIGHTS_DIR = os.path.join(BASE_DIR, 'Weights')
XRAY_NPY_DIR = os.path.join(BASE_DIR, 'IMF_xray_npy')
RUN_NUMBER_FILE = os.path.join(BASE_DIR, 'Weights', 'run_number.txt')


#CNN Hyperparameters

IMG_SIZE = (256,256)        #Other img sizes for vaild padding with 3*3 conv: 320, 512    
FILTER_SHAPE = (3,3,1)      #(height, width, depth) 
NUM_FILTERS = 8             #Number of filters in Conv layer
EPOCHS = 10                #Epochs
LR = 0.001                  #Learning rate

#Seed for reproducibility
np.random.seed(20030701)

''' CNN Layers and Implementations '''

#Base Class for Activation Functions
class Activation:
    def __init__(self, activation, activation_der):
        self.activation = activation
        self.activation_der = activation_der

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient):
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
        self.bias = np.zeros((number_of_filters,1))

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
            self.output[:,:,i] += self.bias[i] 
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
        self.bias -= learning_rate * np.sum(output_gradient, axis=(0, 1)).reshape(-1, 1)
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
    epsilon = 1e-7  
    pred = np.clip(prediction, epsilon, 1 - epsilon) 
    return -np.mean(img_label * np.log(pred) + (1 - img_label) * np.log(1 - pred))
  
#Derivative of Binary Cross Entropy Loss Function
def bce_der(img_label, prediction):
    epsilon = 1e-7 
    pred = np.clip(prediction, epsilon, 1 - epsilon)
    return ((1 - img_label) / (1 - pred) - img_label / pred) / np.size(img_label)

''' TRAINING AND TESTING '''

#Function to train network saves weights and biases for each epoch to a folder
def train():
    #Attempt to load dataset from memory, if not found prepare dataset and save to memory then load
    try:
        train_img, train_lbl = load_train_dataset_from_memory()
    except:
        save_train_dataset(1,IMG_SIZE)    #depth, img_size
        train_img, train_lbl = load_train_dataset_from_memory()

    #Initialize Layers
    conv = Conv(train_img[0].shape,3,NUM_FILTERS)
    relu = ReLU()
    pool = MaxPool()
    dense = Dense(127*127*NUM_FILTERS,1)
    sigmoid = Sigmoid()


    run_dir = create_run_folder()
    

    #Train

    save_weights(run_dir, "conv_init", conv.filters, conv.bias)
    save_weights(run_dir, "dense_init", dense.weights, dense.bias)
    avr_error_log = []
    for epoch in range(EPOCHS):
        epoch_dir = create_epoch_folder(run_dir, epoch)
        epoch_error=0
        train_img, train_lbl = shuffle_dataset(train_img, train_lbl)
        for img, lbl in zip(train_img, train_lbl):

            #Forward Propagation

            conv_output = conv.forward(img)

            relu_output = relu.forward(conv_output)
            
            pool_output = pool.forward(relu_output)
            
            dense_input, conv_shape = Conv_to_Dense(pool_output)
            
            dense_output = dense.forward(dense_input)
            
            sigmoid_output = sigmoid.forward(dense_output)
            
            #Loss Calculation and Log

            epoch_error += bce(lbl, sigmoid_output)  #Binary Cross Entropy Loss to calculate error

            #Backpropagation
            
            loss_gradient = bce_der(lbl, sigmoid_output)
            
            sigmoid_gradient = sigmoid.backward(loss_gradient)
            
            dense_gradient = dense.backward(sigmoid_gradient, LR)
            
            dense_To_conv_gradient = Dense_to_Conv(dense_gradient, conv_shape)
            
            pool_gradient = pool.backward(dense_To_conv_gradient)
            
            relu_gradient = relu.backward(pool_gradient)
            
            conv_gradient = conv.backward(relu_gradient, LR)

        avg_error = epoch_error / len(train_img) 
        print(f"Epoch: {epoch}, Average Error: {avg_error}")

        save_weights(epoch_dir, "conv", conv.filters, conv.bias)
        save_weights(epoch_dir, "dense", dense.weights, dense.bias)
        avr_error_log.append(avg_error)
        if epoch == 0:
            signal.signal(signal.SIGINT, lambda signal, frame: stop_training(signal, frame, run_dir, avr_error_log)) 
    save_mse(run_dir, "error", avr_error_log)
    
#Function to test the network loads weights and biases from input folder
def test():
    pass

''' IMAGE/Dataset Handling '''

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
def save_train_dataset(depth, img_size):
    pneumonia_img, pneumonia_label = preprocess_data(PNEUMONIA_TRAIN_DIR, 1, img_size)
    normal_img, normal_label = preprocess_data(NORMAL_TRAIN_DIR, 0, img_size)

    
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
    
    np.save( os.path.join(XRAY_NPY_DIR, 'train_imgs.npy'), train_imgs)
    np.save( os.path.join(XRAY_NPY_DIR, 'train_lbls.npy'), train_lbls)

# Function to load the training dataset from memory if it exists
def load_train_dataset_from_memory():
    train_imgs = np.load('train_imgs.npy')
    train_lbls = np.load('train_lbls.npy')
    return train_imgs, train_lbls

# Function to shuffle the dataset before each epoch
def shuffle_dataset(train_imgs, train_lbls):
    indices = np.arange(train_imgs.shape[0])
    np.random.shuffle(indices)
    train_imgs = train_imgs[indices]
    train_lbls = train_lbls[indices]
    return train_imgs, train_lbls

'''DATA HANDLIING'''

#Create a new folder for each run to save the weights and biases
def create_run_folder():
    run_num = run_number()
    # Generate a unique folder name for this run
    timestamp = datetime.datetime.now().strftime("%m-%d")
    run_folder = os.path.join(WEIGHTS_DIR, f"RUN_{run_num}__{timestamp}")

    # Create the directory for this run
    os.makedirs(run_folder, exist_ok=True)
    return run_folder

#Create a new folder for each epoch to save the weights and biases
def create_epoch_folder(run_folder, epoch):
    timestamp = datetime.datetime.now().strftime("%H-%M-%S")
    epoch_folder = os.path.join(run_folder, f"Epoch_{epoch}__{timestamp}")
    os.makedirs(epoch_folder, exist_ok=True)
    return epoch_folder

# Save the weights to a folder for each epoch, diffrent file for each layer including weights/filters and biases
def save_weights(epoch_folder,layer_name, weights, biases):

    layer_file = os.path.join(epoch_folder, f"{layer_name}.npz")
    
    # Save both weights and biases in a compressed file
    np.savez(layer_file, weights=weights, biases=biases)

# Load the weights from a folder for each epoch . Layer name is  dense or conv 
def load_weights(run_folder, epoch=None, layer_name=None):

    # Find the latest epoch folder 
    if epoch is None:
        epoch_folders = [f for f in os.listdir(run_folder) if f.startswith('Epoch_')]
        if not epoch_folders:
            raise ValueError("No epoch folders found in the run folder.")
        
        # Sort epoch folders by their modification times
        epoch_folders.sort(key=lambda x: os.path.getmtime(os.path.join(run_folder, x)))
        selected_epoch_folder = epoch_folders[-1]

    # Load the preselected epoch folder
    else:
        selected_epoch_folder = [f for f in os.listdir(run_folder) if f.startswith(f'Epoch_{epoch}__')]
        selected_epoch_folder = selected_epoch_folder[0]
        if not selected_epoch_folder:
            raise ValueError(f"No epoch folder found for epoch {epoch}.")
    
    epoch_folder_path = os.path.join(run_folder, selected_epoch_folder)

    layer_file = os.path.join(epoch_folder_path, f"{layer_name}.npz")
    with np.load(layer_file) as data:
        weights = data["weights"]
        biases = data["biases"]
    return weights, biases

#Save the error for each run a file
def save_mse(run_folder, data_name, data):
    data_file = os.path.join(run_folder, f"{data_name}.txt")

    with open(data_file, 'w') as f:
        # Iterate over the data list and write each entry to the file
        for i, error in enumerate(data):
            f.write(f"Epoch: {i}  Average Error: {error}\n")

#Kepps track of the amount of times the network has been trained for logging purposes
def run_number():
    if os.path.join(RUN_NUMBER_FILE):
        with open(RUN_NUMBER_FILE, 'r') as f:
            run_number = int(f.read())
    else:
        run_number = 0
    
    run_number = run_number + 1

    with open(RUN_NUMBER_FILE, 'w') as f:
        f.write(str(run_number))

    return run_number

#Function to create a README file for each run folder
def create_readme(run_folder):
    readme_file = os.path.join(run_folder, 'README.txt')
    with open(readme_file, 'w') as f:
        f.write(f"Number of Epochs:{EPOCHS}\n")
        f.write(f"Learning Rate: {LR}\n")
        f.write(f"Number of Filters: {NUM_FILTERS}\n")

#Function to stop training And save the error log when ctrl+c is pressed
def stop_training(signal, frame, run_dir, avr_error_log):
    print("Training stopped by user.")
    save_mse(run_dir, "error", avr_error_log)
    sys.exit(0) 

'''TESTING AND DEBUGGING'''

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

#Functtion that loads and prints filters/weights of selected layer, for testing purposes
def print_Filters_1HD(run_folder,epoch=None):
    conv_weights, conv_biases = load_weights(run_folder, epoch, layer_name="conv")
    dense_weights, dense_biases = load_weights(run_folder, epoch, layer_name="dense")
    
    #Print the shape and values of the weights and biases of the conv layer
    print(f"Conv Weights Shape: {conv_weights.shape} \n Conv Filters: {conv_biases.shape}\n")
    print(f"Conv Biases Shape: {conv_biases.shape} \n Conv Biases: {conv_biases}\n")

    #Print the shape and values of the weights and biases of the dense layer
    print(f"Dense Weights Shape: {dense_weights.shape} \n Dense Weights: {dense_weights}\n")
    print(f"Dense Biases Shape: {dense_biases.shape} \n Dense Biases: {dense_biases}\n")
    
#Function to test the forward and backward pass of the network with a single image getting shapes and values of the layers
def test_single_image():
    #Attempt to load dataset from memory, if not found prepare dataset and save to memory then load
    try:
        train_img, train_lbl = load_train_dataset_from_memory()
    except:
        save_train_dataset(1,IMG_SIZE)    #depth, img_size
        train_img, train_lbl = load_train_dataset_from_memory()

    #Initialize Layers
    conv = Conv(train_img[0].shape,3,NUM_FILTERS)
    relu = ReLU()
    pool = MaxPool()
    dense = Dense(127*127*NUM_FILTERS,1)
    sigmoid = Sigmoid()

    #Forward Prop
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

    #Loss Calculation
    loss = bce(train_lbl[0], sigmoid_output)
    print(f"Loss: {loss}")

    #Backpropagation
    loss_gradient = bce_der(train_lbl[0], sigmoid_output)
    print(f"Loss Gradient: {loss_gradient}")

    sigmoid_gradient = sigmoid.backward(loss_gradient)
    print(f"Sigmoid Gradient: {sigmoid_gradient}")

    dense_gradient = dense.backward(sigmoid_gradient, LR)
    print(f"Dense Gradient Shape: {dense_gradient.shape}")

    dense_To_conv_gradient = Dense_to_Conv(dense_gradient, shape)
    print(f"Dense to Conv Gradient Shape: {dense_To_conv_gradient.shape}")

    pool_gradient = pool.backward(dense_To_conv_gradient)
    print(f"Pool Gradient Shape: {pool_gradient.shape}")

    relu_gradient = relu.backward(pool_gradient)
    print(f"Relu Gradient Shape: {relu_gradient.shape}")

    conv_gradient = conv.backward(relu_gradient, LR)
    print(f"Conv Gradient Shape: {conv_gradient.shape}")

#train()

print_Filters_1HD(os.path.join('RUN_1__03-16_14-44-34'))
