/*
* Autograd Engine
*
* Author: Andrew Boessen
*/

#ifndef AUTODIFF_H
#define AUTODIFF_H

#include <cuda_runtime.h>

#include <stdlib.h>

/**
 * @struct Value
 *
 * The `Value` struct is fundamental to the operation of automatic differentiation in the neural network.
 * Each Value instance can be thought of as a node in the computation graph, keeping track of its actual value,
 * gradient, and connections to other nodes (children) it depends on.
 *
 * @param val Actual scalar value represented by this node.
 * @param grad Gradient of the value, computed during the backward pass.
 * @param children Array of pointers to child nodes that this value directly depends on in the computation graph.
 * @param n_children Number of child nodes in the `children` array.
 * @param backward Function pointer to the backward function responsible for computing the gradient for this node.
 */
typedef struct Value {
    float val;  // actual value
    float grad;  // gradient
    struct Value** children;  // children this value depends on
    int n_children;  // number of children
    void (*backward)(struct Value*);  // backward function to compute gradients
} Value;

/**
 * This function allocates memory for a Value object and initializes its attributes.
 */
Value* init_value(float x);

/**
 * This function takes an array of floats and initializes an array of Value objects
 */
Value** init_values(float* arr, size_t len);

/**
 * This function takes two values and returns a new Value object with the sum of the two inputs
 */
Value* add(Value* a, Value* b);

/**
 * This function takes two Value objects and returns new Value object with difference
 */
Value* sub(Value* a, Value* b);

/**
 * This function takes two value objects and multiplies them together and returns new Value with the product
 */
Value* mul(Value* a, Value* b);

/**
 * This function takes two Value objects and devides them and returns a value with the quotient
 */
Value* divide(Value* a, Value* b);

/**
 * This function creates a new Value object that represents one Value object raised to the power of another.
 * The resulting Value object will have a backward function assigned for gradient computation.
 */
Value* power(Value* a, Value* b);

/**
 * Function to calculate gradient of Value object that is a sum
 */
__device__ void add_backwards(Value* v);

/**
 * Function to calculate gradient of Value object that is a difference
 */
__device__ void sub_backwards(Value* v);

/**
 * Computes the gradient of the multiplication operation with respect to its operands.
 */
__device__ void mul_backward(Value* v);

/**
 * Computes the gradient of the division operation with respect to its operands.
 */
__device__ void div_backward(Value* v);

/**
 * Computes the gradient of the power operation with respect to its operands.
 */
void power_backward(Value* v);

/**
 * This function builds a topological order of the computation graph, starting from the given Value object.
 */
void build_topo(Value* v, Value** topo, int* topo_size, Value** visited, int* visited_size);

/**
 * This function outputs the 'val' and 'grad' attributes of the given Value object to the console.
 */
void print_value(Value* v);

/**
 * Recursivly go down children map and output values
 */
void print_children(Value* v);

/**
 * Print the expression of Value v
 */
void print_expression(Value* v);

#endif /* AUTODIFF_H */
