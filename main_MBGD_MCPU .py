import os
import datetime
import time
import gc
import numpy as np
from math import sqrt 
from PIL import Image
from multiprocessing import Pool
from numba import njit

'''
- Mini Batch Gradient Descent(MBGD) CNN Implementation for Pneumonia Detection In Xray Images

- For Sohastic Gradient Descent(SGD) use Batch size 1

- For Batch Gradient Descent(BGD) use Batch size of total number of samples (5216 for cuurent dataset)
'''

#region PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PNEUMONIA_TRAIN_DIR = os.path.join(BASE_DIR, 'IMG_xray', 'train', 'PNEUMONIA')  
NORMAL_TRAIN_DIR = os.path.join(BASE_DIR, 'IMG_xray', 'train', 'NORMAL')

PNEUMONIA_VAL_DIR = os.path.join(BASE_DIR, 'IMG_xray', 'val', 'PNEUMONIA')
NORMAL_VAL_DIR = os.path.join(BASE_DIR, 'IMG_xray', 'val', 'NORMAL')

PNEUMONIA_TEST_DIR = os.path.join(BASE_DIR, 'IMG_xray', 'test', 'PNEUMONIA')
NORMAL_TEST_DIR = os.path.join(BASE_DIR, 'IMG_xray', 'test', 'NORMAL')

WEIGHTS_DIR = os.path.join(BASE_DIR, 'Weights')
XRAY_NPY_DIR = os.path.join(BASE_DIR, 'IMG_xray_npy')
#endregion

#Number of CPU cores available to use for parallelization (16 total on my machine)
CPU_CORES = 8
CHUNKSIZE = 2

#CNN Hyperparameters

IMG_SIZE = (390,390)        # SENSITIVE VALUE for 1HD (256,256) for 2HD (390,390) NEEDS TO BE APPLICABLE TO VALID PADDING AND MAXPOOLING (REQUIRES manual delete of npy files if changed)
FILTER_SHAPE = (3,3,1)      
NUM_FILTERS = [4, 8, 16]         
HIDDEN_LAYERS = 3           
EPOCHS = 100                 
BATCH_SIZE = 16             
LR = 0.001                  

#Amount of training samples needed for weighted loss function due to dataset imbalance 
# (training wont work if diffrent dataset is used unless this is changed)
TRAIN_NORM = 1341
TRAIN_PNEU = 3875
TRAIN_ALL = TRAIN_NORM + TRAIN_PNEU
WEIGHTED_NORM = TRAIN_ALL / (TRAIN_NORM * 2)
WEIGHTED_PNEU = TRAIN_ALL / (TRAIN_PNEU * 2)

#Seed for reproducibility
SEED = 20030701
np.random.seed(SEED)

''' CNN Layers and Implementations '''

#region =================== ACTIVATION FUNCTIONS ==================
#Sigmoid Forward and Backwards with Numba JIT
@njit
def sigmoid_forward(x):
    return 1 / (1 + np.exp(-x))

@njit
def sigmoid_backward(x):
    return x * (1 - x)

#RELU Forward and Backwards with Numba JIT
@njit
def relu_forward(x):
    return np.maximum(0, x)

@njit
def relu_backward(x):
    return (x > 0).astype(np.float32)
#endregion

#region =================== CONVOLUTIONAL LAYER ===================
class Conv:

    def __init__(self, input_shape, filter_size, number_of_filters, pool=None, batch_size=BATCH_SIZE, chunksize=CHUNKSIZE):
        input_height, input_width, input_depth  = input_shape # (height, width, depth)
        self.number_of_filters = number_of_filters
        self.batch_size = batch_size
        self.chunksize = chunksize
        self.pool = pool
        self.input_shape = (batch_size,) + input_shape
        self.input_depth = input_depth
        self.output_shape = (batch_size, input_height - filter_size + 1, input_width - filter_size + 1, number_of_filters) # (height, width, num_filters/depth) 
        self.filters_shape = (number_of_filters, filter_size, filter_size, input_depth) # (num_filters, height, width, depth)
        self.bias = np.zeros((number_of_filters,1), dtype=np.float32)

        #He Initialization for conv layer and Relu activation
        fan_in = filter_size * filter_size * input_depth
        sigma = sqrt(2 / fan_in)
        self.filters = np.random.normal(0, sigma, self.filters_shape).astype(np.float32)

    #Takes batch as input in the shape of (batch, height, width, depth) and returns the output in the shape of (batch, height, width, num_filters)
    def forward_batch(self, input_data):
        self.input = input_data
        try:
            batch_output = self.pool.starmap(conv_forward_single, [(self.input[b], self.filters, self.bias)for b in range(self.batch_size)], chunksize=self.chunksize)
            self.output = np.stack(batch_output, axis=0)
        except Exception as e:
            print("ERROR in conv.forward_batch starmap:", e)
        
        return self.output

    def backward_batch(self, output_gradient, learning_rate):
        try:
            results = self.pool.starmap(conv_backward_single, [(self.input[b], output_gradient[b], self.filters) for b in range(self.batch_size)], chunksize=self.chunksize)
        except Exception as e:
            print("ERROR in conv.backward_batch starmap:", e)
        
        batch_input_gradient = [res[0] for res in results]
        batch_filter_gradient = [res[1] for res in results]

        input_gradient = np.stack(batch_input_gradient, axis=0)
        filter_gradient = sum(batch_filter_gradient)

        self.filters -= learning_rate * (filter_gradient/self.batch_size)
        self.bias -= learning_rate * (np.sum(output_gradient, axis=(0, 1, 2)).reshape(-1, 1) / self.batch_size)

        return input_gradient
 
#Forward pass for single image used in conv class, outside of class for faster cpu parallelization
@njit
def conv_forward_single(input_data, filters, bias):

    number_of_filters, filter_h, filter_w, input_depth = filters.shape
    input_h, input_w, _ = input_data.shape
    output_h = input_h - filter_h + 1
    output_w = input_w - filter_w + 1

    output = np.zeros((output_h, output_w, number_of_filters), dtype=np.float32)

    for f in range(number_of_filters):
        for d in range(input_depth):
            for oh in range(output_h):
                for ow in range(output_w):
                    sum_val = 0.0
                    for fh in range(filter_h):
                        for fw in range(filter_w):
                            sum_val += input_data[oh + fh, ow + fw, d] * filters[f, fh, fw, d]
                    output[oh, ow, f] += sum_val
        output[:, :, f] += bias[f, 0]

    return output

#Backward pass for single image used in conv class, outside of class for faster cpu parallelization
@njit
def conv_backward_single(input_data, output_gradient, filters):
    number_of_filters, filter_height, filter_width, input_depth = filters.shape
    input_height, input_width, _ = input_data.shape
    output_height, output_width, _ = output_gradient.shape

    input_gradient = np.zeros(input_data.shape, dtype=np.float32)
    filter_gradient = np.zeros(filters.shape, dtype=np.float32)

    for f in range(number_of_filters):
        for d in range(input_depth):
            # Filter Gradient valid cross-correlation
            for fh in range(filter_height):
                for fw in range(filter_width):
                    sum_filter = 0.0
                    for oh in range(output_height):
                        for ow in range(output_width):
                            sum_filter += input_data[oh + fh, ow + fw, d] * output_gradient[oh, ow, f]
                    filter_gradient[f, fh, fw, d] += sum_filter
        
            # Input Gradient full convolution
            for ih in range(input_height):
                for iw in range(input_width):
                    sum_input = 0.0
                    for fh in range(filter_height):
                        for fw in range(filter_width):
                            oh = ih - fh
                            ow = iw - fw
                            if 0 <= oh < output_height and 0 <= ow < output_width:
                                sum_input += output_gradient[oh, ow, f] * filters[f, fh, fw, d]
                    input_gradient[ih, iw, d] += sum_input
    
    return input_gradient , filter_gradient
#endregion

#region ======================= DENSE LAYER =======================
class Dense:

    def __init__(self, input_size, output_size, batch_size=BATCH_SIZE):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.bias = np.zeros((1, self.output_size), dtype=np.float32)

        #Initialize weights using Xavier Initialization
        sigma = sqrt(2 / self.input_size)
        self.weights = np.random.normal(0, sigma, (self.output_size, self.input_size)).astype(np.float32)

    def forward(self, input):
        self.input = input.astype(np.float32)
        return dense_forward_njit(self.input, self.weights, self.bias)

    def backward(self, output_gradient, lr):
        output_gradient_fl32 = output_gradient.astype(np.float32)
        input_gradient, weights_update, bias_update = dense_backward_njit(output_gradient_fl32, self.input, self.weights)
        self.weights -= lr * (weights_update/ self.batch_size)
        self.bias -= lr * bias_update
        return input_gradient

#Forward and backward pass compiled with Numba JIT for faster CPU parallelization
@njit
def dense_forward_njit(input_data, weights, bias):
    return input_data @ weights.T + bias

@njit
def dense_backward_njit(output_gradient, input_data, weights):
    output_gradient = output_gradient.astype(np.float32)
    input_data = input_data.astype(np.float32)
    weights = weights.astype(np.float32)

    bias_gradient = np.zeros((1, output_gradient.shape[1]), dtype=np.float32)
    for j in range(output_gradient.shape[1]):
        bias_gradient[0, j] = np.sum(output_gradient[:, j]) / output_gradient.shape[0]
    
    input_gradient = output_gradient @ weights
    weights_gradient = output_gradient.T @ input_data

    return input_gradient, weights_gradient, bias_gradient
#endregion

#region ====================== MAXPOOL LAYER ======================
class MaxPool:
    def __init__(self, pool=None, pool_size=2, stride=2, batch_size=BATCH_SIZE, chunksize=CHUNKSIZE):
        self.pool_size = pool_size
        self.pool = pool
        self.stride = stride
        self.batch_size = batch_size
        self.chunksize = chunksize

    def forward_batch(self, input_data):
        self.input = input_data
        try:
            batch_output = self.pool.starmap(pool_forward_single, [(input_data[b], self.pool_size, self.stride) for b in range(self.batch_size)], chunksize=self.chunksize)
        except Exception as e:
            print("ERROR in pool.forward_batch starmap:", e)
        
        self.output = np.stack(batch_output, axis=0)
        return self.output

    def backward_batch(self,output_gradient):
        try:
            batch_input_gradient = self.pool.starmap(pool_backward_single, [(self.input[b], output_gradient[b], self.pool_size, self.stride) for b in range(self.batch_size)], chunksize=self.chunksize)
        except Exception as e:
            print("ERROR in pool.backward_batch starmap:", e)
        
        return np.stack(batch_input_gradient, axis=0)

#Forward pass for single image used in maxpool class, outside of class and compiled with njit for faster cpu parallelization
@njit
def pool_forward_single(input_data, pool_size, stride):
    input_height, input_width, input_depth = input_data.shape
    output_height = (input_height - pool_size) // stride + 1
    output_width = (input_width - pool_size) // stride + 1
    output = np.zeros((output_height, output_width, input_depth), dtype=np.float32)
        
    for d in range(input_depth):
        for h in range(output_height):
            height_start = h * stride
            for w in range(output_width):
                width_start = w * stride

                max_val = -np.inf
                for i in range(pool_size):
                    for j in range(pool_size):
                        region = input_data[height_start + i, width_start + j, d]
                        if region > max_val:
                            max_val = region
                output[h, w, d] = max_val

    return output

#Backward pass for single image used in maxpool class, outside of class for faster cpu parallelization
@njit
def pool_backward_single(input_data, output_gradient, pool_size, stride):
    input_height, input_width, input_depth = input_data.shape
    output_height, output_width, _ = output_gradient.shape
    input_gradient = np.zeros(input_data.shape, dtype=np.float32)

    for d in range(input_depth):
        for h in range(output_height):
            height_start = h * stride
            for w in range(output_width):
                width_start = w * stride
                max_val = -np.inf
                max_i = 0
                max_j = 0

                for i in range(pool_size):
                    for j in range(pool_size):
                        val = input_data[height_start + i, width_start + j, d]
                        if val > max_val:
                            max_val = val
                            max_i = i
                            max_j = j
                
                input_gradient[height_start + max_i, width_start + max_j, d] = output_gradient[h, w, d]

    return input_gradient
#endregion

#region ====================== LOSS FUNCTIONS =====================
#Binary Cross Entropy Loss Function njit compiled
@njit
def weighted_bce(img_label, prediction, weighted_norm, weighted_pneu):
    batch_label = img_label.reshape(-1, 1)
    epsilon = 1e-7  
    pred = np.clip(prediction, epsilon, 1 - epsilon)
    loss = -(weighted_pneu * batch_label * np.log(pred) + weighted_norm * (1 - batch_label) * np.log(1 - pred))
    return np.mean(loss)
  
#Derivative of Binary Cross Entropy Loss Function njit compiled
@njit
def weighted_bce_der(img_label, prediction, weighted_norm, weighted_pneu):
    batch_label = img_label.reshape(-1, 1)
    epsilon = 1e-7 
    pred = np.clip(prediction, epsilon, 1 - epsilon)
    der = -(weighted_pneu * (batch_label / pred)) + (weighted_norm * ((1 - batch_label) / (1 - pred)))
    return der / np.size(batch_label)
#endregion

#Batch normalization njit compiled
@njit
def batch_norm(batch):
    epsilon=1e-5
    batch_size, height, width, depth = batch.shape
    output = np.zeros((batch_size, height, width, depth), dtype=np.float32)
    for d in range(depth):
        depth_value = batch[:, :, :, d].ravel()
        mean = np.mean(depth_value)
        var = np.var(depth_value)
        std = np.sqrt(var + epsilon)
        for b in range(batch_size):
            for h in range(height):
                for w in range(width):
                    output[b, h, w, d] = (batch[b, h, w, d] - mean) / std
    return output

#Get the shapes to intialize conv layers
def get_Conv_Shapes(img_shape, filter_size=3, stride=1, padding=0, pool_size=2, pool_stride=2, num_filters=NUM_FILTERS, hidden_layers=HIDDEN_LAYERS):
    conv_shapes = []
    h, w, d = img_shape
    for layer in range(hidden_layers):
        conv_shapes.append((h, w, d))
            
        # Calculate the shape after convolution
        h = (h - filter_size + 2 * padding) // stride + 1
        w = (w - filter_size + 2 * padding) // stride + 1
        d = num_filters[layer]
        
        # Calculate the shape after pooling
        h = (h - pool_size) // pool_stride + 1
        w = (w - pool_size) // pool_stride + 1
    
    conv_shapes.append((h, w, d))
    return conv_shapes

'''TRAIN VAL TEST'''

#Function to train network, saves weights and biases for each epoch to a folder
def train(run_file=None, img_size=IMG_SIZE, hidden_layers=HIDDEN_LAYERS, num_filters=NUM_FILTERS, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, weighted_norm=WEIGHTED_NORM, weighted_pneu=WEIGHTED_PNEU):
    validate_dir()

    #Attempt to load dataset from memory, if not found prepare dataset and save to memory then load
    try:
        train_img, train_lbl = load_train_dataset_from_memory()
    except:
        save_train_dataset(1,img_size)    #depth, img_size
        train_img, train_lbl = load_train_dataset_from_memory()

    #Initialize Layers
    conv_shapes = get_Conv_Shapes(train_img[0].shape)
    dense_shape = conv_shapes[-1]
    dense_init_input = 1
    for i in dense_shape:
        dense_init_input *= i

    conv_layers = []
    pool_layers = []
    for h in range(hidden_layers):
        conv_layers.append(Conv(conv_shapes[h],3, num_filters[h],))
        pool_layers.append(MaxPool())
    dense = Dense(dense_init_input,1)

    #Load weights and biases from preselected run folder if given to continue training
    if run_file is not None:
        run_dir = run_file
        last_epoch = run_number()
        avr_error_log = load_log(run_file)
        for h in range(hidden_layers):
            conv_layers[h].filters, conv_layers[h].bias = load_weights(run_file, layer_name=f"conv{h}")
        dense.weights, dense.bias = load_weights(run_file, layer_name="dense")
    else:
        avr_error_log = []
        last_epoch = 0
        run_dir = create_run_folder()
    
    #print init weights and biases for debugging
    #for i, conv in enumerate(conv_layers):
    #    print(f"conv{i}:{conv.filters}")
    #    print(f"bias{i}:{conv.bias}")
    #print(f"dense:{dense.weights}")
    #print(f"bias:{dense.bias}")

    #Train
    for epoch in range(epochs):

        #Initialize Pool for parallelization Per Epoch
        with Pool(CPU_CORES) as pool:
            for conv in conv_layers:
                conv.pool = pool
            for pool_layer in pool_layers:
                pool_layer.pool = pool

            epoch_error=0
            train_img, train_lbl = shuffle_dataset(train_img, train_lbl)
            num_samples = len(train_img)
            num_batches = num_samples // batch_size
            for batch_index in range(num_batches):

                #Prepare batches for Each Epoch
                start = batch_index * batch_size
                end = start + batch_size
                batch_img = train_img[start:end]
                batch_lbl = train_lbl[start:end]
            
                #Forward Propagation
                hidden_layer = batch_img
                for h in range(hidden_layers):

                    conv_output = conv_layers[h].forward_batch(hidden_layer)  #I:(batch, height, width, input_depth)  O:(batch, height, width, num_filters)
                    
                    norm_output = batch_norm(conv_output)  #I:(batch, height, width, num_filters)  O:(batch, height, width, num_filters)

                    relu_output = relu_forward(norm_output)   #I:(batch, height, width, num_filters)  O:(batch, height, width, num_filters)
                    
                    pool_output = pool_layers[h].forward_batch(relu_output)   #I:(batch, height, width, num_filters)  O:(batch, height, width, num_filters)
                    
                    hidden_layer = pool_output
            
                conv_shape = hidden_layer.shape
                dense_input = hidden_layer.reshape(batch_size, -1)      #I:(batch, height, width, num_filters)  O:(batch, Flattened_Img) 

                dense_output = dense.forward(dense_input)               #I:(batch, Flattened_Img)  O:(batch, 1)
        
                sigmoid_output = sigmoid_forward(dense_output)          #I:(batch, 1)  O:(batch, 1)

                #Binary Cross Entropy Loss Calculation
                epoch_error += weighted_bce(batch_lbl, sigmoid_output, weighted_norm, weighted_pneu)           #I:(batch, 1)  O:(batch, 1)
               
                #Backpropagation
            
                loss_gradient = weighted_bce_der(batch_lbl, sigmoid_output, weighted_norm, weighted_pneu)      #I:(batch, 1)  O:(batch, 1)
               
                sigmoid_gradient = sigmoid_backward(sigmoid_output)      #I:(batch, 1)  O:(batch, 1)
                
                dense_gradient_input = loss_gradient * sigmoid_gradient 
                dense_gradient = dense.backward(dense_gradient_input, lr)   #I:(batch, 1)  O:(batch, Flattened_Img) 
                
                dense_To_conv_gradient= dense_gradient.reshape(conv_shape)   #I:(batch, Flattened_Img)  O:(batch, height, width, num_filters)
            
                hidden_layer_back = dense_To_conv_gradient

                for h in reversed(range(hidden_layers)):
                    pool_gradient = pool_layers[h].backward_batch(hidden_layer_back)   #I:(batch, height, width, num_filters)  O:(batch, height, width, num_filters)
                    
                    relu_gradient = relu_backward(pool_gradient)     #I:(batch, height, width, num_filters)  O:(batch, height, width, num_filters)
                    
                    conv_gradient = conv_layers[h].backward_batch(relu_gradient, lr)
                    
                    hidden_layer_back = conv_gradient

                #Clear memory
                del conv_output, norm_output,relu_output, pool_output, hidden_layer, conv_shape, dense_input, dense_output, sigmoid_output, loss_gradient, sigmoid_gradient,dense_gradient_input, dense_gradient, dense_To_conv_gradient, pool_gradient, relu_gradient, conv_gradient, hidden_layer_back
                
                for conv in conv_layers:
                    conv.input = None
                    conv.output = None
                for pool in pool_layers:
                    pool.input = None
                    pool.output = None
                
                gc.collect()

        #Print and Save Epoch Results
        avg_error = epoch_error / num_batches
        print(f"Epoch: {epoch+1+last_epoch}, Average Error: {avg_error}")

        epoch_dir = create_epoch_folder(run_dir, epoch)
        for h in range(hidden_layers):
            save_weights(epoch_dir, f"conv{h+1}", conv_layers[h].filters, conv_layers[h].bias)
        save_weights(epoch_dir, "dense", dense.weights, dense.bias)
        avr_error_log.append(avg_error)
        save_mse(run_dir, avr_error_log) 

#Function to evaluate the network in each epoch
def eval():
    pass

#Function to test the network loads weights and biases from input folder
def test():
    pass

''' IMAGE/Dataset Handling '''

# Function to preprocess the images : Grayscale -> Resize -> Normalize -> Return as NumPy array
def preprocess_data(directory, label, IMG_SIZE=IMG_SIZE):
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
def save_train_dataset(depth, img_size, pneumonia_train_dir=PNEUMONIA_TRAIN_DIR, normal_train_dir=NORMAL_TRAIN_DIR, xray_npy_dir=XRAY_NPY_DIR):
    pneumonia_img, pneumonia_label = preprocess_data(pneumonia_train_dir, 1, img_size)
    normal_img, normal_label = preprocess_data(normal_train_dir, 0, img_size)

    
    train_imgs = np.concatenate((pneumonia_img, normal_img), axis=0)
    train_lbls = np.concatenate((pneumonia_label, normal_label), axis=0)
    indices = np.arange(train_imgs.shape[0])
    np.random.shuffle(indices)
    train_imgs = train_imgs[indices]
    train_lbls = train_lbls[indices]

    # Reshape for CNN input (number of Images, height, width, depth)
    train_imgs = train_imgs.reshape(train_imgs.shape[0], img_size[0], img_size[1], depth)

    print(f"Processed dataset shape: {train_imgs.shape}, Labels shape: {train_lbls.shape}")
    
    np.save( os.path.join(xray_npy_dir, 'train_imgs.npy'), train_imgs)
    np.save( os.path.join(xray_npy_dir, 'train_lbls.npy'), train_lbls)

# Function to load the training dataset from memory if it exists
def load_train_dataset_from_memory(xray_npy_dir=XRAY_NPY_DIR):
    train_imgs = np.load(os.path.join(xray_npy_dir,'train_imgs.npy')).astype(np.float32)
    train_lbls = np.load(os.path.join(xray_npy_dir, 'train_lbls.npy')).astype(np.float32)
    return train_imgs, train_lbls

# Function to shuffle the dataset before each epoch
def shuffle_dataset(train_imgs, train_lbls):
    indices = np.arange(train_imgs.shape[0])
    np.random.shuffle(indices)
    train_imgs = train_imgs[indices]
    train_lbls = train_lbls[indices]
    return train_imgs, train_lbls

'''DATA SAVE/LOAD'''

#Checks if required files and directories exist, if not creates them
def validate_dir(pneumonia_train_dir=PNEUMONIA_TRAIN_DIR, normal_train_dir=NORMAL_TRAIN_DIR, pneumonia_test_dir=PNEUMONIA_TEST_DIR, normal_test_dir=NORMAL_TEST_DIR, weights_dir=WEIGHTS_DIR, xray_npy_dir=XRAY_NPY_DIR):
    if not os.path.exists(pneumonia_test_dir) or not os.path.exists(normal_test_dir) or not os.path.exists(pneumonia_train_dir) or not os.path.exists(normal_train_dir):
        print("Error: Dataset not found.") 
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    if not os.path.exists(xray_npy_dir):
        os.makedirs(xray_npy_dir)

#Create a new folder for each run to save the weights and biases
def create_run_folder(weight_dir=WEIGHTS_DIR):
    run_num = run_number()
    # Generate a unique folder name for this run
    timestamp = datetime.datetime.now().strftime("%m-%d")
    run_folder = os.path.join(weight_dir, f"RUN_{run_num}__{timestamp}")

    # Create the directory for this run
    os.makedirs(run_folder, exist_ok=True)
    create_readme(run_folder)

    return run_folder

#Create a new folder for each epoch to save the weights and biases
def create_epoch_folder(run_folder, epoch):
    timestamp = datetime.datetime.now().strftime("%H-%M-%S")
    epoch_folder = os.path.join(run_folder, f"Epoch_{epoch+1}__{timestamp}")
    os.makedirs(epoch_folder, exist_ok=True)
    return epoch_folder

# Save the weights to a folder for each epoch, diffrent file for each layer including weights/filters and biases
def save_weights(epoch_folder,layer_name, weights, biases):

    layer_file = os.path.join(epoch_folder, f"{layer_name}.npz")
    
    # Save both weights and biases in a compressed file
    np.savez(layer_file, weights=weights, biases=biases)

# Load the weights from a folder for each epoch . Layer name is  dense or conv 
def load_weights(run_folder, epoch=None, layer_name=None):
    run_folder = os.path.join(WEIGHTS_DIR, run_folder)
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
def save_mse(run_folder, data):
    data_file = os.path.join(run_folder, "errorLog.txt")

    with open(data_file, 'w') as f:
        # Iterate over the data list and write each entry to the file
        for i, error in enumerate(data):
            f.write(f"Epoch: {i+1}  Average Error: {error}\n")

#loads error log from previous trainig runs to continue error file correctly
def load_log(run_folder):
    avg_error_log = []
    data_file = os.path.join(run_folder, "errorLog.txt")

    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5 and parts[0] == 'Epoch:':
                error = float(parts[4])
                avg_error_log.append(error)

    return avg_error_log

#Kepps track of the amount of times the network has been trained for logging purposes
def run_number(weight_dir=WEIGHTS_DIR):
    run_numbers = []
    for filename in os.listdir(weight_dir):
        if filename.startswith('RUN_'):
            parts = filename.split('_')
            if len(parts) >= 2:
                number_str = parts[1]
                try:
                    run_num = int(number_str)
                    run_numbers.append(run_num)
                except ValueError:
                    print(f"Invalid run number found: {number_str}")
                    pass
    
    
    current_max = max(run_numbers) if run_numbers else 0
    next_run = current_max + 1
    
    return next_run

#Function to create a README file for each run folder
def create_readme(run_folder, seed=SEED, hidden_layers=HIDDEN_LAYERS, batch_size=BATCH_SIZE, lr=LR, num_filters=NUM_FILTERS, img_size=IMG_SIZE, filter_shape=FILTER_SHAPE, cpu_cores=CPU_CORES, chunksize=CHUNKSIZE):
    timestamp = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
    readme_file = os.path.join(run_folder, 'README.txt')
    with open(readme_file, 'w') as f:
        f.write(f"Run Start: {timestamp}\n")
        f.write(f"CPU cores used: {cpu_cores}\n")
        f.write(f"Chunksize: {chunksize}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"HIDDEN_LAYERS: {hidden_layers}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Number of Filters: {num_filters}\n")
        f.write(f"IMG_SIZE: {img_size}\n")
        f.write(f"FILTER_SHAPE: {filter_shape}\n")
        
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

def log_batch_conv_values(batch_idx, conv_outputs):
    for i, output in enumerate(conv_outputs):
        mean = np.mean(output)
        std = np.std(output)
        relu_ratio = np.mean(output > 0)
        print(f"Batch {batch_idx} | Conv Layer {i} | mean: {mean:.4f}, std: {std:.4f}, relu_ratio: {relu_ratio:.4f}")

#Functtion that loads and prints filters/weights of selected layer, for testing purposes
def print_filters(run_folder,epoch=None, hidden_layers=HIDDEN_LAYERS):
    conv_weights = []
    conv_biases = []
    for h in range(hidden_layers):
        conv_weights, conv_biases = load_weights(run_folder, epoch, layer_name=f"conv{h+1}")
        print(f"Conv{h+1} Weights Shape: {conv_weights.shape} \n Conv{h+1} Filters: {conv_weights}\n")
        print(f"Conv{h+1} Biases Shape: {conv_biases.shape} \n Conv{h+1} Biases: {conv_biases}\n")

    dense_weights, dense_biases = load_weights(run_folder, epoch, layer_name="dense")
    print(f"Dense Weights Shape: {dense_weights.shape} \n Dense Weights: {dense_weights}\n")
    print(f"Dense Biases Shape: {dense_biases.shape} \n Dense Biases: {dense_biases}\n")

''' EXECUTION '''

def main():
    start_time = time.time()
    try:
        train()

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    except Exception as e:
        print("Error during training:", e) 

    finally:
        end_time = time.time()
        time_taken = end_time - start_time
        mins, secs = divmod(time_taken, 60)
        print(f"\nTraining completed in {int(mins)} min {int(secs)} sec")

if __name__ == "__main__":
    main()
   
