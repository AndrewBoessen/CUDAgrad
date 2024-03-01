/*
* Autograd Engine
*
* Author: Andrew Boessen
*/

#ifndef AUTODIFF_H
#define AUTODIFF_H

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
 * This function outputs the 'val' and 'grad' attributes of the given Value object to the console.
 */
void print_value(Value* v);

#endif /* AUTODIFF_H */