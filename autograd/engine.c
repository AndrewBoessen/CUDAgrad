/*
* Implementation of autograd engine 
*
* Author: Andrew Boessen
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "engine.h"

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
   out->backward = NULL;
   return out;
}

/**
 * This function takes two Value objects and returns new Value object with difference
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
    out->backward = NULL;
    return out;
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
 * Recursivly go down children map and output values
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

    printf("Value(val=%.2f, grad=%.2f)\n", v->val, v->grad);
}