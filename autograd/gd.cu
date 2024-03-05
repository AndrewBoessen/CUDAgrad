/*
* Implement gradient descent part of autograd engine in CUDA
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
#include "engine.h"
}

extern "C" {
void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}

/**
 * This helper function allocates new memory for a specified amout of Values.
 *
 * @param v (return paramter) The pointer to the start of the Values in memory
 * @param num Number of values to allocate
 */
void allocValue(Value* v, size_t num) {
    cudaError_t err = cudaMallocManaged(&v, num * sizeof(Value));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed while allocating Value: %s\n", cudaGetErrorString(err));
        // Handle the error appropriately
        exit(1);
    }
    // Prefetch memory to correct devices (e.g. CPU or GPU)
    cudaMemPrefetchAsync(v, num * sizeof(Value), MAIN_DEVICE);
}

/**
 * This helper function allocates new memory for an array of Values.
 *
 * @param prt (return parameter) Pointer to start of list of Values
 * @param len Length of array of Value
 */
void allocValueArr(Value** ptr, size_t len) {
    cudaError_t err = cudaMallocManaged(&ptr, len * sizeof(Value*));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed while allocating array of Values: %s\n", cudaGetErrorString(err));
        // Handle the error appropriately
        exit(1);
    }
    // Prefetch memory to correct devices (e.g. CPU or GPU)
    cudaMemPrefetchAsync(ptr, len * sizeof(Value*), MAIN_DEVICE);

}

/**
 * Function to calculate gradient of Value object that is a sum
 * 
 * Computes gradient with respect to the operands
 *
 * @param v Pointer to the Value object resulting from addition
 */
__device__ void add_backwards(Value* v) {
    v->children[0]->grad += v->grad;
    v->children[1]->grad += v->grad;
}

/**
 * Function to calculate gradient of Value object that is a difference
 *
 * Computes the gradient with respect to the operands
 *
 * @param v Pointer to Value object resulting from subtraction
 *
 * @note
 * The final gradient for the operand is its local gradient multiplied by any external gradient flowing from a parent.
 * The local derivative for the subtraction is:
 *     dv/da (locally) = 1
 *     dv/db (locally) = -1
 * The external gradient (from parent nodes) is stored in v->grad.
 * Thus, the final gradient for a is: dv/da = 1 * v->grad
 * And for b is: dv/db = -1 * v->grad
 */
__device__ void sub_backwards(Value* v) {
    v->children[0]->grad += v->grad;
    v->children[1]->grad -= v->grad;
}

/**
 * Computes the gradient of the multiplication operation with respect to its operands.
 *
 * @param v Pointer to the Value object resulting from the multiplication.
 *
 * @note
 * The final gradient for the operand is its local gradient multiplied by any external gradient flowing from a parent.
 * The local derivative for the multiplication is:
 *     dv/da (locally) = b
 *     dv/db (locally) = a
 * The external gradient (from parent nodes) is stored in v->grad.
 * Thus, the final gradient for a is: dv/da = b * v->grad
 * And for b is: dv/db = a * v->grad
 */
__device__ void mul_backward(Value* v) {
    // printf("child %.f grad = %f*%f", v->children[0], v->children[1]->val, v->grad);
    // printf("child %.f grad = %f*%f", v->children[1], v->children[0]->val, v->grad);
    v->children[0]->grad += v->children[1]->val * v->grad;
    v->children[1]->grad += v->children[0]->val * v->grad;
}

/**
 * Computes the gradient of the division operation with respect to its operands.
 *
 * @param v Pointer to the Value object resulting from the division.
 *
 * @note
 * The final gradient for the operand is its local gradient multiplied by any external gradient flowing from a parent.
 * The local derivative for the division is:
 *     dv/da (locally) = 1/b
 *     dv/db (locally) = -a/(b^2)
 * The external gradient (from parent nodes) is stored in v->grad.
 * Thus, the final gradient for a is: dv/da = (1/b) * v->grad
 * And for b is: dv/db = (-a/(b^2)) * v->grad
 */
__device__ void div_backward(Value* v) {
    v->children[0]->grad += (1.0 / v->children[1]->val) * v->grad;
    v->children[1]->grad += (-v->children[0]->val / (v->children[1]->val * v->children[1]->val)) * v->grad;
}

/**
 * Computes the gradient of the power operation with respect to its operands.
 *
 * @param v Pointer to the Value object resulting from the power operation.
 *
 * @note
 * The final gradient for the operand is its local gradient multiplied by any external gradient flowing from a parent.
 * The local derivative for the power operation is:
 *     dv/da (locally) = b * a^(b-1)
 *     dv/db (locally) = a^b * log(a)
 * The external gradient (from parent nodes) is stored in v->grad.
 * Thus, the final gradient for a is: dv/da = (b * a^(b-1)) * v->grad
 * And for b is: dv/db = (v * log(a)) * v->grad
 */
__device__ void power_backward(Value* v) {
    v->children[0]->grad += (v->children[1]->val * pow(v->children[0]->val, v->children[1]->val - 1)) * v->grad;
    if (v->children[0]->val > 0) {  // Ensure base is positive before computing log
        v->children[1]->grad += (log(v->children[0]->val) * pow(v->children[0]->val, v->children[1]->val)) * v->grad;
    }
}

/**
 * This helper function doubles the capacity of array. It does this by allocating a new
 * array with double the capacity, copying the existing data to the new array,
 * and updating the topo pointer to point to the new array.
 *
 * @param arr Pointer to the pointer that holds the array.
 * @param arr_size Pointer to the variable that holds the current size of the array.
 * @param arr_capacity Pointer to the variable that holds the current capacity of the array.
 */
void resize_array(Value*** arr, int* arr_size, int* arr_capacity) {
    *arr_capacity *= 2;
    Value** new_arr = (Value**)realloc(*arr, *arr_capacity * sizeof(Value*));
    if (new_arr == NULL) {
        printf("Memory allocation failed.\n");
        exit(1);
    }
    *arr = new_arr;
}

/**
 * Helper function to build a topological order of the computation graph, starting from the given Value object.
 *
 * @param v The starting Value object for the topological sort.
 * @param topo A pointer to an array where the topological order will be stored.
 * @param topo_size Pointer to the size of the topo array.
 * @param visited Pointer to an array that keeps track of visited Value objects.
 * @param visited_size Pointer to the size of the visited array.
 */
void build_topo(Value* v, Value*** topo, int* topo_size, int* topo_capacity, Value*** visited, int* visited_size, int* visited_capacity) {
    for (int i = 0; i < *visited_size; ++i) {
        if ((*visited)[i] == v) return;
    }

    if (*visited_size == *visited_capacity) {
        resize_array(visited, visited_size, visited_capacity);
    }
    (*visited)[*visited_size] = v;
    (*visited_size)++;

    for (int i = 0; i < v->n_children; ++i) {
        build_topo(v->children[i], topo, topo_size, topo_capacity, visited, visited_size, visited_capacity);
    }

    if (*topo_size == *topo_capacity) {
        resize_array(topo, topo_size, topo_capacity);
    }
    (*topo)[*topo_size] = v;
    (*topo_size)++;
}
}