# CUDAgrad

Inspired by [@karpathy's - micrograd](https://github.com/karpathy/micrograd).

![Gradient Descent](./gd.jpg)

An autograd engine is the technical implementation of backpropogation algorithm that allows neural nets to learn.
This is a implementation of a very lightweight autograd engine using C and CUDA for gpu acceleration.

## Dependencies

- CUDA toolkit
- C compiler (e.g., GCC, Clang)

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/AndrewBoessen/CUDAgrad
   ```

2. Navigate to the project directory:

   ```bash
   cd CUDAgrad
   ```

3. Build the project:

   ```bash
   make
   ```

## Example Usage

Here is a example of how to create a simple expression with the autograd engine

```c
#include <stdio.h>
#include <stdlib.h>

#include "engine.h"

int main(int argc, char** argv) {
    // Test add and subtract
    Value* x = init_value(1.0);
    Value* y = init_value(1.5);

    Value* sum = add(x,y);

    Value* t = init_value(3.0);

    // Z = (X + Y) - T
    Value* z = sub(sum, t);

    // Outputs expression in postfix notation:
    // X Y + T - = Z
    print_expression(z);

    return EXIT_SUCCESS;
}
```

Output:

```bash
1.00 1.50 + 3.00 - = -0.50
```

## Neural Network

![Neural Network](./nn.jpg)

Using the nueral network library, it is easy to create a MLP and train on it.

The library uses GPU acceleration for the forward pass and for training.

### Example Usage

Simple MLP with 3 hidden layers

```c
#include "nn.h"
#include "engine.h"

int main() {
    int n_inputs = 2;
    int n_outputs = 2;

    int sizes[] = {n_inputs, 5, 10, 5, n_outputs};
    int nlayers = sizeof(sizes) / sizeof(int);

    MLP* mlp = init_mlp(sizes, nlayers);
    show_params(mlp);

    // Allocate inputs
    Value** in;
    allocValueArr(&in, n_inputs);
    // Set inputs
    in[0] = init_value(1.0);
    in[1] = init_value(1.0);

    Value** out = mlp_forward(mlp, in, n_inputs);

    print_value(out[0]);
    print_value(out[1]);
}
```

### GPU Acceleration

1. Forward Pass

   On the forward pass each layer's neurons are computed in parallel.
   There are $n$ blocks and $i$ threads, where $n$ is the number of neurons and $i$ is $n \cdot \text{number of inputs}$.
   The layer's outputs are fed into the next layer's inputs.

2. Stochastic Gradient Descent

   Utilizing GPU parallelism involves running independent gradient descents at the block level and mini-batch gradient descent at the thread level. Each thread samples k points, and after each iteration, blocks update their gradients by averaging thread-level estimates. Finally, the parameter estimates are obtained by averaging the estimates from all blocks.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
