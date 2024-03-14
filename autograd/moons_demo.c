/*
 * Train MLP for make moons classification
 */
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#include "nn.h"
#include "engine.h"
#include "data.h"

#define EPOCHS 20
#define BATCH_SIZE 25
#define LEARNING_RATE 0.01
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
                //printf("GT: %f OUT: %f\n", gt[0]->val, out[0]->val);
                // Calculate loss for single datapoint
                Value* loss = mse_loss(out, gt, NUM_OUTPUTS);

                total_loss = add(total_loss, loss);
            }
            // Do backprop on total loss of batch
            backward(total_loss);
            // Single step after batch
            update_weights(mlp, LEARNING_RATE);
            // Zero grads for next batch
            zero_grad(mlp);

            // Print loss
            epoch_loss += total_loss->val;
            
        }
        printf("EPOCH: %d LOSS: %f\n", i+1, epoch_loss/(DATA_SIZE/BATCH_SIZE));
    }

    return EXIT_SUCCESS;
}
