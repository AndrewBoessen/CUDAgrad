/*
* Implementation of autograd engine 
*
* Author: Andrew Boessen
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "engine.h"
#include "gd.cuh"

/**
 * This function allocates memory for a Value object and initializes its attributes.
 *
 * @param x The float value to initialize the Value object with.
 * @return A pointer to the newly created Value object.
 */
Value* init_value(float x) {
    Value* v = (Value*)malloc(sizeof(Value));
    v->val = x;
    v->grad = 0;
    v->children = NULL;
    v->n_children = 0;
    v->backward = NULL;
    return v;
}

/**
 * This function takes an array of floats and initializes an array of Value objects
 *
 * @param arr The array of floats to become Value objects
 * @param len The length of the input array
 */
 Value** init_values(float* arr, size_t len) {
    // Allocate memory for Values
    Value** values = (Value**)malloc(len * sizeof(Value*));

    if (values == NULL) {
        perror("Memory allocation for values failed");
        exit(1);
    }

    for (size_t i = 0; i < len; i++) {
        values[i] = init_value(arr[i]);
    }

    return values;
 }

/**
 * This function takes two values and returns a new Value object with the sum of the two inputs
 *
 * @param a forst value to add
 * @param b Second value to add
 * @return A pointer to the new Value object representing the sum.
 */
Value* add(Value* a, Value* b) {
   Value* out = (Value*)malloc(sizeof(Value));
   out->val = a->val + b->val;
   out->grad = 0;
   // Allocate memory for children
   out->children = (Value**)malloc(2 * sizeof(Value*));
   // Set children to pointers of a and b
   out->children[0] = a;
   out->children[1] = b;
   out->n_children = 2;
   out->backward = add_backwards;
   return out;
}

/**
 * This function takes two Value objects and returns new Value object with difference
 *
 * @param a Value to subtract
 * @param b Value to subtract
 * @return A pointer to the new Value object representing the difference.
 */
Value* sub(Value* a, Value* b) {
    Value* out = (Value*)malloc(sizeof(Value));
    out->val = a->val - b->val;
    out->grad = 0;
    // Allocate memory for children
    out->children = (Value**)malloc(2 * sizeof(Value*));
    // Set children
    out->children[0] = a;
    out->children[1] = b;
    out->n_children = 2;
    out->backward = sub_backwards;
    return out;
}

/**
 * This function takes two value objects and multiplies them together and returns new Value with the product
 *
 * @param a Value to multiply
 * @param b Value to multiply by
 * @return A pointer to the new Value object representing the product.
 */
Value* mul(Value* a, Value* b) {
    Value* out = (Value*)malloc(sizeof(Value));
    out->val = a->val * b->val;
    out->grad = 0;
    // Allocate memory for children
    out->children = (Value**)malloc(2 * sizeof(Value*));
    // Set children
    out->children[0] = a;
    out->children[1] = b;
    out->n_children = 2;
    out->backward = mul_backward;
    return out;
}

/**
 * This function takes two Value objects and devides them and returns a value with the quotient
 *
 * @param a Pointer to the numerator Value object.
 * @param b Pointer to the denominator Value object.
 * @return A pointer to the new Value object representing the quotient.
 */
Value* divide(Value* a, Value* b) {
    if(b->val == 0.0) {
        printf("Error: Division by zero\n");
        exit(1);
    }

    Value* out = (Value*)malloc(sizeof(Value));
    out->val = a->val / b->val;
    out->grad = 0;
    out->children = (Value**)malloc(2 * sizeof(Value*));
    out->children[0] = a;
    out->children[1] = b;
    out->n_children = 2;
    out->backward = div_backward;
    return out;
}

/**
 * This function creates a new Value object that represents one Value object raised to the power of another.
 * The resulting Value object will have a backward function assigned for gradient computation.
 *
 * @param base Pointer to the base Value object.
 * @param exponent Pointer to the exponent Value object.
 * @return A pointer to the new Value object representing the power result.
 */
Value* power(Value* a, Value* b) {
    Value* out = (Value*)malloc(sizeof(Value));
    out->val = pow(a->val, b->val);
    out->grad = 0;
    out->children = (Value**)malloc(2 * sizeof(Value*));
    out->children[0] = a;
    out->children[1] = b;
    out->n_children = 2;
    out->backward = power_backward;
    return out;
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
    Value** new_arr = realloc(*arr, *arr_capacity * sizeof(Value*));
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


/**
 * This function outputs the 'val' and 'grad' attributes of the given Value object to the console.
 *
 * @param v Pointer to the Value object to be printed.
 */
void print_value(Value* v) {
    printf("Value(val=%.2f, grad=%.2f)\n", v->val, v->grad);
}

/**
 * Recursivly go down children map and output values. This prints the expression of v in postfix notation
 * 
 * @param v Pointer to starting Value in graph
 */
void print_children(Value *v) {
    // Go down graph until node has no children
    // Recursivly print childrens' values
    if (v->n_children == 1) {
        print_children(v->children[0]);
    } else if (v->n_children == 2) {
        print_children(v->children[0]);
        print_children(v->children[1]);
    }

    char operand;
    if (v->backward == add_backwards) {
        operand = '+';
    } else if (v->backward == sub_backwards) {
        operand = '-';
    } else if (v->backward == mul_backward) {
        operand = '*';
    } else if (v->backward == div_backward) {
        operand = '/';
    } else {
        operand = ' ';
    }

    if (v->n_children == 0) {
        printf("%.2f ", v->val);
    } else {
        printf("%c ", operand);
    }
}

/**
 * Print the expression of Value v
 */
void print_expression(Value* v) {
    print_children(v);
    printf("= %.2f\n", v->val);
}