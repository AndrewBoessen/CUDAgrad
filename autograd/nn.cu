/*
* Implementation of nueral network lib with GPU acceleration
*
* Author: Andrew Boessen
*/

#include <stddef.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand_kernel.h>

extern "C" {
#include "nn.h"
#include "engine.h"
}

extern "C"{
__device__ void mul_dev(Value* w, Value* x, Value* v) {
    v->val = w->val * x->val;
    v->grad = 0;
    v->children[0] = w;
    v->children[1] = x;
    v->n_children = 2;
    v->op = MUL;
}

__device__ void add_dev(Value* out, Value* b, Value* v) {
    v->val = out->val + b->val;
    v->grad = 0;
    v->children[0] = out;
    v->children[1] = b;
    v->n_children = 2;
    v->op = ADD;
}

__device__ void relu_dev(Value* out, Value* v) {
    v->val = (out->val < 0) ? 0 : out->val;
    v->grad = 0;
    v->children[0] = out;
    v->n_children = 1;
    v->op = RELU;
}

/**
 * This helper function allocates new memory for a specified amount of Neurons.
 *
 * @param n (return parameter) The pointer to the start of the Neurons in memory
 */
void allocNeuron(Neuron** n) {
    cudaError_t err = cudaMalloc(n, sizeof(Neuron));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed while allocating Neuron: %s\n", cudaGetErrorString(err));
        // Handle the error appropriately
        exit(1);
    }
}

/**
 * This helper function allocates new memory for an array of Neurons.
 *
 * @param ptr (return parameter) Pointer to start of list of Neurons
 * @param len Length of array of Neurons
 */
void allocNeuronArr(Neuron*** ptr, size_t len) {
    cudaError_t err = cudaMalloc(ptr, len * sizeof(Neuron*));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed while allocating array of Neurons: %s\n", cudaGetErrorString(err));
        // Handle the error appropriately
        exit(1);
    }
}

/**
 * @brief Initialize a neuron with random weights and zero bias.
 *
 * @param nin Number of input connections.
 * @param nonlin Activation function flag (1 for ReLU, 0 for linear).
 * @return Pointer to the initialized Neuron.
 */
Neuron* init_neuron(int nin, int nonlin) {
    Neuron* neuron;
    allocNeuron(&neuron);
    cudaMalloc(&(neuron->w), nin);
    for (int i = 0; i < nin; i++) {
        // random values between -1 and 1
        cudaMemcpy(neuron->w[i], init_value((rand() % 2000 - 1000) / 1000.0), sizeof(Value), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(&neuron->b, init_value(0), sizeof(Value), cudaMemcpyHostToDevice);
    cudaMemcpy(&neuron->nin, &nin, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&neuron->nonlin, &nonlin, sizeof(int), cudaMemcpyHostToDevice);;
    return neuron;
}

/**
 * @brief Initialize a neural network layer with specified neurons.
 *
 * @param nin Number of input connections for each neuron.
 * @param nout Number of neurons in the layer.
 * @param nonlin Activation function flag for all neurons (1 for ReLU, 0 for linear).
 * @return Pointer to the initialized Layer.
 */
Layer* init_layer(int nin, int nout, int nonlin) {
    Layer* layer;
    // Allocate one layer in memory
    cudaMalloc(&layer, sizeof(Layer));
    // Allocate neurons that make up layer
    allocNeuronArr(&(layer->neurons), nout);
    for (int i = 0; i < nout; i++) {
        // Init neurons
        cudaMemcpy(layer->neurons[i], init_neuron(nin, nonlin), sizeof(Neuron), cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(&layer->nout, &nout, sizeof(int), cudaMemcpyHostToDevice);
    return layer;
}

/**
 * @brief Perform forward pass computation for a layer.
 *
 * @param layer Pointer to the layer.
 * @param x Array of input values for the layer.
 * @param out Array of values corresponding to output of layer.
 * @param products Array of Values for products of inputs and weights.
 * @param biases Values to store sum of output ands bias
 * @param activations Values to store outputs activation function
 */
__global__ void layer_forward(Layer* layer, Value** x, Value** out, Value** products, Value** biases, Value** activations) {
    // Id of datapoint in batch
    int datapoint_id = blockIdx.y;

    // Index of neuron to computer (block)
    int neuron_idx = blockIdx.x;
    // Current neuron in layer
    Neuron* n = layer->neurons[neuron_idx];

    // Index of cuurent input of neuron
    int input_idx = blockDim.x * blockIdx.y + threadIdx.x;

    // Index of product within array
    int prod_idx = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;

    // Index of nuerons output
    int out_idx = datapoint_id * gridDim.x + neuron_idx;

    // Set paramters of product
    Value* prod = products[prod_idx];
    mul_dev(n->w[threadIdx.x], x[input_idx], prod);

    // Add product to children of neuron output
    out[out_idx]->children[threadIdx.x] = prod;
    // Update neuron output value
    atomicAdd(&(out[out_idx]->val), prod->val);

    // Wait for all thread to finish computing products
    __syncthreads();

    // Add bias to sum and activate if nonlin
    // Only run if last thread in block
    if (threadIdx.x == blockDim.x - 1) {
        Value* sum = biases[out_idx];
        add_dev(out[out_idx], n->b, sum);
        
        out[out_idx] = sum;

        if (n->nonlin) {
            // Activate with ReLU function if nonlin
            Value* relu_val = activations[out_idx];
            relu_dev(out[out_idx], relu_val);

            out[out_idx] = relu_val;
        }
    }
}

/**
 * @brief Initialize a Multilayer Perceptron (MLP) with the specified layer sizes.
 *
 * @param sizes Array of layer sizes, where each element represents the number of neurons in that layer.
 * @param nlayers Number of layers in the MLP.
 * @return Pointer to the initialized MLP.
 */
MLP* init_mlp(int* sizes, int nlayers) {
    // Allocate memory for MLP
    MLP* mlp;
    cudaMalloc(&mlp, sizeof(MLP));
    // Allocate space for layers in MLP
    cudaMalloc(&(mlp->layers), (nlayers - 1) * sizeof(Layer*));
    for (int i = 0; i < nlayers - 1; i++) {
        int nonlin = (i != nlayers - 2);  // nonlinearity for all layers except the last one
        mlp->layers[i] = init_layer(sizes[i], sizes[i+1], nonlin);
    }
    mlp->nlayers = nlayers - 1;
    return mlp;
}

/**
 * @brief Perform forward pass computation for the entire MLP.
 *
 * @param mlp Pointer to the MLP.
 * @param x Array of input values for the MLP.
 * @param nin Number of inputs
 * @return Array of output values from the final layer of the MLP.
 */
Value** mlp_forward(MLP* mlp, Value** in, int nin) {
    // Copy input array to device
    Value** x;
    cudaMalloc(&x, nin * sizeof(Value*));
    for (int i = 0; i < nin; i++) {
        cudaMalloc(&x[i], sizeof(Value));
        cudaMemcpy(&x[i], in[i], sizeof(Value), cudaMemcpyDeviceToHost);
    }

    for (int i = 0; i < mlp->nlayers; i++) {
        Layer* curr_layer = mlp->layers[i];

        int curr_layer_out;
        cudaMemcpy(&curr_layer_out, &curr_layer->nout, sizeof(int), cudaMemcpyDeviceToHost);

        Value** out;
        cudaMalloc(&out, curr_layer_out * sizeof(Value*));
        for(int i = 0; i < curr_layer_out; i++) {
            cudaMalloc(&out[i]->children, nin * sizeof(Value*));
        }

        Value** products;
        cudaMalloc(&products, nin * curr_layer_out * sizeof(Value*));
        for(int i = 0; i < nin * curr_layer_out; i++) {
            cudaMalloc(&out[i]->children, 2 * sizeof(Value*));
        }

        // Grid size: single datapoint so y is 1
        dim3 grid_size(curr_layer->nout, 1);
        layer_forward<<<grid_size, nin>>>(curr_layer, x, out, products, out, out);
        // Wait for kernel to finish before updating x
        cudaDeviceSynchronize();
        // Number of next inputs are number of current outputs
        nin = curr_layer->nout;
        // Next layers inputs are current layers outputs
        x = out;
    }
    
    // Copy outputs to host
    Value** out= (Value**)malloc(nin * sizeof(Value*));
    for (int i = 0; i < nin; i++) {
        cudaMemcpy(&out[i], &x[i], sizeof(Value), cudaMemcpyDeviceToHost);
    }
    return out;
}

/**
 * @brief Helper function to free arrays of allocated Value in maanged memory
 *
 * @param arr Array of Value arrs to free
 */
void freePtrArr(Value*** arrs, int len) {
    for (int i = 0; i < len; i++) {
        Value** curr_arr = arrs[i];
        // Loop until NULL pointer encountered
        for (int j = 0; curr_arr[j] != NULL; j++) {
            Value* curr_val = curr_arr[j];
            cudaError_t err_c = cudaFree(curr_val->children);
            cudaError_t err = cudaFree(curr_val);
        }
        cudaFree(curr_arr);
    } 
}

/**
 * @brief Train the MLP for one batch
 *
 * Do a forward pass for an entire batch of data points,
 * then do a backward pass to find gradients and update paramters
 *
 * @param mlp MLP object to train
 * @param x inputs for the batch
 * @param nin number of neurons in input layer
 * @param y_true ground truth for datapoints in batch
 * @param lr learning rate
 * @param batch_size number of datapoints in batch
 * @return Total loss of entire batch
 */
/*
float train(MLP* mlp, Value** x, int nin, Value** y_true, float lr, int batch_size){
    // Arrays for storing Value arrays to later be freed
    Value** products_ptrs[mlp->nlayers];
    Value** bias_ptrs[mlp->nlayers];
    Value** act_ptrs[mlp->nlayers];
    Value** sum_ptrs[mlp->nlayers];

    for (int l = 0; l < mlp->nlayers; l++) {
        Layer* curr_layer = mlp->layers[l];
        // Total number of neurons in entire batch
        int total_neurons = curr_layer->nout * batch_size;

        // Grid dimensions: x for neurons in layer, y for batch size
        dim3 grid_size(curr_layer->nout, batch_size);
        layer_forward<<<grid_size, nin>>>(curr_layer, x, out, products, biases, activations);
        // Wait for kernel to finish before updating x
        cudaDeviceSynchronize();
        // Number of next inputs are number of current outputs
        nin = curr_layer->nout;
        // Next layers inputs are current layers outputs
        x = out;

        // Add Value arrs to arrays to free
        products_ptrs[l] = products;
        bias_ptrs[l] = biases;
        act_ptrs[l] = activations;
        sum_ptrs[l] = sums;
    }
    // Calculate loss for each output
    Value* total_loss = init_value(0.0);

    for (int i = 0; i < batch_size * nin; i+=nin) {
        Value* curr_data_out[nin];
        Value* curr_data_gt[nin];
        // Populate array with slice from output
        for (int j = 0; j < nin; j++) {
            curr_data_out[j] = x[i + j];
            curr_data_gt[j] = y_true[i + j];
        }
        // Calculate loss for each datapoint in batch
        Value* loss = mse_loss(curr_data_out, curr_data_gt, nin);

        // Add datapoint loss to total loss
        total_loss = add(total_loss, loss);
    }
    // Do backprop to find gradients
    backward(total_loss);
    // Single step for batch
    update_weights(mlp, lr);
    // zero grads before next batch
    zero_grad(mlp);
    
    // Free network from memory
    freePtrArr(sum_ptrs, mlp->nlayers);
    freePtrArr(products_ptrs, mlp->nlayers);
    freePtrArr(bias_ptrs, mlp->nlayers);
    freePtrArr(act_ptrs, mlp->nlayers);

    return total_loss->val;
}
*/

/**
 * @brief Compute the mean squared error (MSE) loss between predicted and true values.
 *
 * @param y_pred Array of predicted values.
 * @param y_true Array of true values.
 * @param size Number of values in y_pred and y_true arrays.
 * @return Pointer to the computed MSE loss value.
 */
Value* mse_loss(Value** y_pred, Value** y_true, int size) {
    
    Value* loss = init_value(0.0);
    for (int i = 0; i < size; i++) {
        Value* diff = sub(y_pred[i], y_true[i]);
        Value* sq = power(diff, init_value(2.0));
        loss = add(loss, sq);
    }
    loss = divide(loss, init_value(size));

    return loss;
}

/**
 * @brief CUDA kernel to zero weight and bias of mlp
 *
 * @param layers Layers of the MLP to update
 */
__global__ void zero_grad_kernel(Layer** layers) {
    // Id for layer
    int layer_idx = blockIdx.x;
    Layer* l = layers[layer_idx];

    // Id for neuron within id
    int neuron_idx = blockIdx.y % l->nout;
    Neuron* n = l->neurons[neuron_idx];

    // Get param to update from thread id
    int weight_idx = threadIdx.x % n->nin;
    // Zero grads for weight and bias
    n->w[weight_idx]->grad = 0;
    n->b->grad = 0;
    
}

/**
 * @brief Host function to zero gradients for params in MLP
 *
 * @param mlp MLP to zero grads
 */
void zero_grad(MLP* mlp) {
    // Get maximum layer size
    int max_neurons = 0;
    for (int i = 0; i < mlp->nlayers; i++) {
        if (mlp->layers[i]->nout > max_neurons) {
            max_neurons = mlp->layers[i]->nout;
        }
    }
    // Dimensions of grid
    // X dim is for layers
    // Y dim is for neurons in layer
    dim3 grid_size(mlp->nlayers, max_neurons);
    // Call Kernel to zero params' grads
    zero_grad_kernel<<<grid_size,max_neurons>>>(mlp->layers);
    // Wait for kernel to finish
    cudaDeviceSynchronize();
}

/**
 * @brief Device func to update the weights of a value using gradient descent.
 *
 * @param v Pointer to the value whose weights need to be updated.
 * @param lr Learning rate for the weight update.
 */
__device__ __inline__ void update_weights_dev(Value* v, float lr) {
    v->val -= lr * v->grad;
}

/**
 * @brief CUDA kernel to update paramaters of MLP
 *
 * @param layers Layers of the MLP to update
 * @param lr Learning rate
 */
__global__ void update_params(Layer** layers, float lr) {
    // Id for layer
    int layer_idx = blockIdx.x;
    Layer* l = layers[layer_idx];

    // Id for neuron within id
    int neuron_idx = blockIdx.y;
    if (neuron_idx <= l->nout - 1) {
        Neuron* n = l->neurons[neuron_idx];

        // Get param to update from thread id
        int weight_idx = threadIdx.x;

        if (weight_idx <= n->nin - 1) {
            update_weights_dev(n->w[weight_idx], lr);
        }
        if(weight_idx == n->nin -1) {
            update_weights_dev(n->b, lr);
        }
    }
}

/**
 * @brief Host function to update weight
 *
 * @param mlp MLP to update weights for
 */
void update_weights(MLP* mlp, float lr) {
    // Get maximum layer size
    int max_neurons = 0;
    for (int i = 0; i < mlp->nlayers; i++) {
        if (mlp->layers[i]->nout > max_neurons) {
            max_neurons = mlp->layers[i]->nout;
        }
    }
    // Dimensions of grid
    // X dim is for layers
    // Y dim is for neurons in layer
    dim3 grid_size(mlp->nlayers, max_neurons);

    update_params<<<grid_size, max_neurons>>>(mlp->layers, lr);

    cudaDeviceSynchronize();
}

/**
 * @brief Display the parameters (weights and biases) of the MLP.
 *
 * @param mlp Pointer to the MLP.
 */
void show_params(MLP* mlp){
    printf("\nMLP\n");
    for (int i = 0; i < mlp->nlayers; i++) {
        Layer* layer = mlp->layers[i];
        printf("\nLayer%i:\n", i);
        for (int j = 0; j < layer->nout; j++) {
            Neuron* neuron = layer->neurons[j];
            for (int k = 0; k < neuron->nin; k++) {
                print_value(neuron->w[k]);
            }
        }
    }
    printf("\n\n");
}

/**
 * @brief Free the memory allocated for a neuron.
 *
 * @param neuron Pointer to the neuron to be freed.
 */
void free_neuron(Neuron* neuron) {
    for (int i = 0; i < neuron->nin; i++) {
        free_value(neuron->w[i]);
    }
    cudaFree(neuron->w);
    free_value(neuron->b);
    cudaFree(neuron);
}

/**
 * @brief Free the memory allocated for a layer.
 *
 * @param layer Pointer to the layer to be freed.
 */
void free_layer(Layer* layer) {
    for (int i = 0; i < layer->nout; i++) {
        free_neuron(layer->neurons[i]);
    }
    cudaFree(layer->neurons);
    cudaFree(layer);
}

/**
 * @brief Free the memory allocated for the entire MLP.
 *
 * @param mlp Pointer to the MLP to be freed.
 */
void free_mlp(MLP* mlp) {
    for (int i = 0; i < mlp->nlayers; i++) {
        free_layer(mlp->layers[i]);
    }
    cudaFree(mlp->layers);
    cudaFree(mlp);
}
}
