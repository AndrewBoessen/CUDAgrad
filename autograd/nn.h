/*
* Tiny Nueral Network Library
*
* Author: Andrew Boessen
*/

#include "engine.h"

/**
 * @struct Neuron
 * @brief Represents a single neuron in a neural network layer.
 *
 * A neuron has weights corresponding to each input, a bias term, and an activation function.
 */

typedef struct Neuron {
    Value** w;  // array of weights
    Value* b;   // bias
    int nin;    // number of input neurons
    int nonlin; // nonlinearity flag: 1 for ReLU, 0 for linear
} Neuron;

/**
 * @struct Layer
 * @brief Represents a single layer in the neural network.
 *
 * A layer consists of multiple neurons.
 */
typedef struct Layer {
    Neuron** neurons;  // array of neurons
    int nout;          // number of output neurons
} Layer;

/**
 * @struct MLP
 * @brief Represents a Multilayer Perceptron (MLP) neural network.
 *
 * An MLP consists of multiple layers.
 */
typedef struct MLP {
    Layer** layers;  // array of layers
    int nlayers;     // number of layers
} MLP;

Neuron* init_neuron(int nin, int lonlin);

Layer* init_layer(int nin, int nout, int nonlin);

MLP* init_mlp(int* sizes, int nlayers);

Value** mlp_forward(MLP* mlp, Value** x, int nin);

Value* mse_loss(Value** y_pred, Value** y_true, int size);

void update_weights(MLP* mlp, float lr);

void show_params(MLP* mlp);

void zero_grad(MLP* mlp);

float train(MLP* mlp, Value** x, int nin, Value** y_true, float lr, int batch_size);

void free_neuron(Neuron* neuron);

void free_layer(Layer* layer);

void free_mlp(MLP* mlp);
