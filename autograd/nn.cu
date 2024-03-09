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
    v->grad = 0;
    v->children[0] = w;
    v->children[1] = x;
    v->n_children = 2;
     v->val = w->val * x->val;
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
 * @param num Number of neurons to allocate
 */
void allocNeuron(Neuron** n, size_t num) {
    cudaError_t err = cudaMallocManaged(n, num * sizeof(Neuron));
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
    cudaError_t err = cudaMallocManaged(ptr, len * sizeof(Neuron*));
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
    allocNeuron(&neuron, 1);
    allocValueArr(&(neuron->w), nin);
    for (int i = 0; i < nin; i++) {
        neuron->w[i] = init_value((rand() % 2000 - 1000) / 1000.0);  // random values between -1 and 1
    }
    neuron->b = init_value(0);
    neuron->nin = nin;
    neuron->nonlin = nonlin;
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
    cudaMallocManaged(&layer, sizeof(Layer));
    // Allocate neurons that make up layer
    allocNeuronArr(&(layer->neurons), nout);
    for (int i = 0; i < nout; i++) {
        // Init neurons
        layer->neurons[i] = init_neuron(nin, nonlin);
    }
    layer->nout = nout;
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
    // Index of neuron to computer (block)
    int neuron_idx = blockIdx.x;

    // Current neuron in layer
    Neuron* n = layer->neurons[neuron_idx];

    // Index of cuurent input of neuron
    int input_idx = threadIdx.x % blockDim.x;

    // Index of product within array
    int prod_idx = input_idx + blockIdx.x * blockDim.x;

    // Set paramters of product
    Value* prod = products[prod_idx];
    mul_dev(n->w[input_idx], x[input_idx], prod);

    // Add product to children of neuron output
    out[neuron_idx]->children[input_idx] = prod;
    // Update neuron output value
    atomicAdd(&(out[neuron_idx]->val), prod->val);

    // Wait for all thread to finish computing products
    __syncthreads();

    // Add bias to sum and activate if nonlin
    // Only run if last thread in block
    if (input_idx == blockDim.x - 1) {
        Value* sum = biases[neuron_idx];
        add_dev(out[neuron_idx], n->b, sum);
        
        out[neuron_idx] = sum;

        if (n->nonlin) {
            // Activate with ReLU function if nonlin
            Value* relu_val = activations[neuron_idx];
            relu_dev(out[neuron_idx], relu_val);

            out[neuron_idx] = relu_val;
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
    cudaMallocManaged(&mlp, sizeof(MLP));
    // Allocate space for layers in MLP
    cudaMallocManaged(&(mlp->layers), (nlayers - 1) * sizeof(Layer*));
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
Value** mlp_forward(MLP* mlp, Value** x, int nin) {
    for (int i = 0; i < mlp->nlayers; i++) {
        Layer* curr_layer = mlp->layers[i];

        // Allocate empty value arr for outputs
        float initialSums[curr_layer->nout];
        memset(initialSums, 0.0, curr_layer->nout * sizeof(float));
        // Initialize sums to 0.0
        Value** out = init_values(initialSums, curr_layer->nout);
        // Allocate space for children of outputs
        for(int i = 0; i < curr_layer->nout; i++) {
            allocValueArr(&(out[i]->children), nin);
            out[i]->n_children = nin;
            out[i]->op = ADD;
        }

        // Allocate array for prodcuts of inputs and weights
        Value** products;
        allocValueArr(&products, nin * curr_layer->nout);
        // Allocate space for products children
        for(int i = 0; i < nin * curr_layer->nout; i++) {
            products[i] = init_value(0);
            allocValueArr(&(products[i]->children), 2);
        }

        // Allocate Values to store sum of output ands bias
        Value** biases;
        allocValueArr(&biases, curr_layer->nout);
        for(int i = 0; i < curr_layer->nout; i++) {
            biases[i] = init_value(0);
            allocValueArr(&(biases[i]->children), 2);
        }

        // Allocate Value to store outputs activation function
        Value** activations;
        allocValueArr(&activations, curr_layer->nout);
        for(int i = 0; i < curr_layer->nout; i++) {
            activations[i] = init_value(0);
            allocValueArr(&(activations[i]->children), 1);
        }

        layer_forward<<<curr_layer->nout, nin * curr_layer->nout>>>(curr_layer, x, out, products, biases, activations);
        // Wait for kernel to finish before updating x
        cudaDeviceSynchronize();
        // Number of next inputs are number of current outputs
        nin = curr_layer->nout;
        // Next layers inputs are current layers outputs
        x = out;
    }
    return x;
}

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
 * @brief Update the weights of a value using gradient descent.
 *
 * @param v Pointer to the value whose weights need to be updated.
 * @param lr Learning rate for the weight update.
 */
__device__ void update_weights(Value* v, float lr) {
    v->val -= lr * v->grad;
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
