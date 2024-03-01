/*
* Implementation of autograd engine 
*
* Author: Andrew Boessen
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
 * This function outputs the 'val' and 'grad' attributes of the given Value object to the console.
 *
 * @param v Pointer to the Value object to be printed.
 */
void print_value(Value* v) {
    printf("Value(val=%.2f, grad=%.2f)\n", v->val, v->grad);
}