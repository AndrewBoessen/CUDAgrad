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
        // Variable learning rate
        float lr = LEARNING_RATE - (0.009 * ((float)(i+1)/EPOCHS));

        shuffle_entries(entries, DATA_SIZE);
        // SGD - calculate loss for a batch of 10 data points
        for (int j = 0; j < DATA_SIZE / BATCH_SIZE; j++) {
            // starting index
            int starting_idx = j * BATCH_SIZE;
            // Batch data and ground truth
            float inputs[NUM_INPUTS * BATCH_SIZE];
            float grnd_truth[NUM_OUTPUTS * BATCH_SIZE];
            // Select next 10 unvisited datapoints in shuffled array
            for (int n = starting_idx; n < starting_idx + BATCH_SIZE; n++) {
                Entry curr_entry = entries[n];
                // add to bath inputs
                inputs[n * NUM_INPUTS] = curr_entry.x;
                inputs[n * NUM_INPUTS + 1] = curr_entry.y;
                
                // Expected y
                grnd_truth[n * NUM_OUTPUTS] = curr_entry.label;
            }
            Value** in = init_values(inputs, NUM_INPUTS * BATCH_SIZE);
            Value** gt = init_values(grnd_truth, NUM_OUTPUTS * BATCH_SIZE);

            // Train batch
            Value** out = train(mlp, in, NUM_INPUTS, gt, lr, BATCH_SIZE);
        }
        printf("EPOCH: %d\n", i+1);
    }

    return EXIT_SUCCESS;
}
