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
 *
 * @param a forst value to add
 * @param b Second value to add
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
 * Function to calculate gradient of Value object that is a sum
 * 
 * Computes gradient with respect to the operands
 *
 * @param v Pointer to the Value object resulting from addition
 */
void add_backwards(Value* v) {
    v->children[0]->grad += v->grad;
    v->children[1]->grad += v->grad;
}

/**
 * Function to calculate gradient of Value object that is a difference
 *
 * Computes the gradient with respect to the operands
 *
 * @param v Pointer to Value object resulting from subtraction
 */
void sub_backwards(Value* v) {
    v->children[0]->grad += v->grad;
    v->children[1]->grad -= v->grad;
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