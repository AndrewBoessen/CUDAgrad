/*
* CUDA Gradient Descent
*
* Author: Andrew Boessen
*
*/

#ifndef GRAD_H
#define GRAD_H

#include <cuda_runtime.h>

void printCudaInfo();

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
__device__ void power_backward(Value* v);

/**
 * Function to create a list of nodes in topological order in CUDA memory
 */
Value* create_topological_list_cuda(Value* root);

#endif