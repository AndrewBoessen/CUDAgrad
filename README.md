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

    // Define the sizes of the layers
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

    // Compute outputs with forward pass
    Value** out = mlp_forward(mlp, in, n_inputs);

    print_value(out[0]);
    print_value(out[1]);
}
```

### Demo - Make Moons

This is an example traning a MLP for classification on the [make moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) dataset

![data](./moons.png)

This example uses a MLP with an input layer with two neurons for X and Y cordinates, 2 hidden layers of 16 neurons and a single output neuron for binary classification.

To train, use stochastic gradient descent. In this example, the network is trained for 15 epochs with a batch size of 10. This implements a variable learning rate that decreases with epoch index.

```c
/*
 * Train MLP for make moons classification
 */
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#include "nn.h"
#include "engine.h"
#include "data.h"

#define EPOCHS 15
#define BATCH_SIZE 10
#define LEARNING_RATE 0.001
#define DATA_SIZE 1000
#define NUM_INPUTS 2
#define NUM_OUTPUTS 1

/**
 * Swaps two Entry structs in the array.
 */
void swap_entries(Entry* a, Entry* b) {
    Entry temp = *a;
    *a = *b;
    *b = temp;
}

/**
 * Shuffles the array of Entry structs using the Fisher-Yates shuffle algorithm.
 */
void shuffle_entries(Entry* entries, int count) {
    for (int i = count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        swap_entries(&entries[i], &entries[j]);
    }
}

int main() {
    // Seed the random number generator
    srand(time(NULL));
    
    // Load data from data file
    Entry entries[DATA_SIZE];
    const char *filename = "./data/make_moons.csv";
    int num_entries = load_data(filename, entries);

    if (num_entries == -1) {
        fprintf(stderr, "Failed to load data from file %s\n", filename);
        return EXIT_FAILURE;
    }

    printf("Loaded %d entries from %s\n", num_entries, filename);

    // Init MLP
    int sizes[] = {NUM_INPUTS, 16, 16, NUM_OUTPUTS};
    int nlayers = sizeof(sizes) / sizeof(int);

    MLP* mlp = init_mlp(sizes, nlayers);

    printf("Training for %d Epochs with Batch Size %d\n", EPOCHS, BATCH_SIZE);
    // Train for number of epochs
    for (int i = 0; i < EPOCHS; i++) {
        // Variable learning rate
        float lr = 0.01 - (0.009 * ((float)(i+1)/EPOCHS));

        float epoch_loss = 0.0;

        shuffle_entries(entries, DATA_SIZE);
        // SGD - calculate loss for a batch of 10 data points
        for (int j = 0; j < DATA_SIZE / BATCH_SIZE; j++) {
            // zero loss for batch
            Value* total_loss = init_value(0.0);
            // starting index
            int starting_idx = j * BATCH_SIZE;
            // Select next 10 unvisited datapoints in shuffled array
            for (int n = starting_idx; n < starting_idx + BATCH_SIZE; n++) {
                Entry curr_entry = entries[n];
                // Alloc new input array
                float inputs[NUM_INPUTS] = {curr_entry.x, curr_entry.y};
                Value** in = init_values(inputs, NUM_INPUTS);
                // Expected y
                float outputs[NUM_OUTPUTS] = {curr_entry.label};
                Value** gt = init_values(outputs, NUM_OUTPUTS);

                // Forward pass for single datapoint
                Value** out = mlp_forward(mlp, in, NUM_INPUTS);

                // Calculate loss for single datapoint
                Value* loss = mse_loss(out, gt, NUM_OUTPUTS);

                total_loss = add(total_loss, loss);
            }
            // Do backprop on total loss of batch
            backward(total_loss);
            // Single step after batch
            update_weights(mlp, lr);
            // zero grads for next batch
            zero_grad(mlp); 

            // Print loss
            epoch_loss += total_loss->val;
            //printf("EPOCH: %d LOSS: %f\n", i+1, total_loss->val);
        }
        printf("EPOCH: %d LOSS: %f\n", i+1, epoch_loss/DATA_SIZE);
    }

    return EXIT_SUCCESS;
}
```

Output:

```
Loaded 1000 entries from ./data/make_moons.csv
Training for 15 Epochs with Batch Size 10
EPOCH: 1 LOSS: 0.144856
EPOCH: 2 LOSS: 0.029270
EPOCH: 3 LOSS: 0.017946
EPOCH: 4 LOSS: 0.015143
EPOCH: 5 LOSS: 0.012307
EPOCH: 6 LOSS: 0.011319
EPOCH: 7 LOSS: 0.009980
EPOCH: 8 LOSS: 0.009541
EPOCH: 9 LOSS: 0.009046
EPOCH: 10 LOSS: 0.008184
EPOCH: 11 LOSS: 0.007889
EPOCH: 12 LOSS: 0.007456
EPOCH: 13 LOSS: 0.007199
EPOCH: 14 LOSS: 0.006906
EPOCH: 15 LOSS: 0.006717
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
