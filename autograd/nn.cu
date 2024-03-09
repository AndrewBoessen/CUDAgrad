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
    Value** weights;
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
 * @return Array of output values from all neurons in the layer.
 */
__global__ void layer_forward(Layer* layer, Value** x, Value** out) {
    // Index of neuron to computer (block)
    int neuron_idx = blockIdx.x;
    // Current neuron in layer
    Neuron* n = layer->neurons[neuron_idx];

    // Index of cuurent input of neuron
    int input_idx = threadIdx.x % blockDim.x;
    // Product of input for neuron and weight
    Value* prod = mul(n->w[input_idx], x[input_idx]);

    // Update the output value with new product
    // Atomic update to not interfere with other threads
    atomicExch(&out[neuron_idx], add(out[neuron_idx], prod));

    // Wait for all thread to finish computing products
    __syncthreads();

    // Add bias to sum and activate if nonlin
    // Only run if last thread in block
    if (input_idx == blockDim.x - 1) {
        out[neuron_idx] = add(out[neuron_idx], n->b);
        if (n->nonlin) {
            out[neuron_idx] = relu(out[neuron_idx]);
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
 * @return Array of output values from the final layer of the MLP.
 */
__global__ Value** mlp_forward(MLP* mlp, Value** x) {
    for (int i = 0; i < mlp->nlayers; i++) {
        x = layer_forward(mlp->layers[i], x);
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
    
    Value* loss = make_value(0.0);
    for (int i = 0; i < size; i++) {
        Value* diff = sub(y_pred[i], y_true[i]);
        Value* sq = power(diff, make_value(2.0));
        loss = add(loss, sq);
    }
    loss = divide(loss, make_value(size));

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
}