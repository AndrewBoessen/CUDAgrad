/*
 * Train MLP for make moons classification
 */
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

#include "nn.h"
#include "engine.h"
#include "data.h"

#define EPOCHS 15
#define BATCH_SIZE 10
#define LEARNING_RATE 0.01
#define DATA_SIZE 1000
#define TRAIN_SIZE 900
#define TEST_SIZE 100
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
    int sizes[] = {NUM_INPUTS, 5, 15, 5, NUM_OUTPUTS};
    int nlayers = sizeof(sizes) / sizeof(int);

    MLP* mlp = init_mlp(sizes, nlayers);

    // Batch data and ground truth
    float* inputs = (float*)malloc(NUM_INPUTS * BATCH_SIZE * sizeof(float));
    float* grnd_truth = (float*)malloc(NUM_OUTPUTS * BATCH_SIZE * sizeof(float));

    // Shuffle entries before taking training dataset
    shuffle_entries(entries, DATA_SIZE);

    int total_neurons = 0;
    int total_params = 0;
    for (int i = 1; i < nlayers; i++) {
        total_neurons += sizes[i];
        total_params += sizes[i-1] * sizes[i];
    }
    // Allocate empty value arr for outputs
    Value** out;
    allocValueArr(&out, BATCH_SIZE * total_neurons);
    // Allocate space for children of outputs
    int j = 0;
    for (int i = 0; i < nlayers - 1; i++) {
        int curr_size = sizes[i];
        for (int k = 0; k < curr_size * BATCH_SIZE; k++, j++) {
            out[j] = init_value(0.0);
            allocValueArr(&out[j]->children, curr_size + 1);
            out[j]->n_children = curr_size + 1;
            out[j]->op = ADD;
        }
    }
    
    // Allocate array for prodcuts of inputs and weights
    Value** products;
    allocValueArr(&products, BATCH_SIZE * total_params);
    // Allocate space for products children
    for(int i = 0; i < BATCH_SIZE * total_params; i++) {
        products[i] = init_value(0);
        allocValueArr(&(products[i]->children), 2);
    }

    printf("Training for %d Epochs with Batch Size %d\n", EPOCHS, BATCH_SIZE);
    // Train for number of epochs
    for (int i = 0; i < EPOCHS; i++) {
        float epoch_loss = 0.0;

        // Variable learning rate
        float lr = LEARNING_RATE - (0.009 * ((float)(i+1)/EPOCHS));

        // Only train on training set
        shuffle_entries(entries, TRAIN_SIZE);

        // SGD - calculate loss for a batch of 10 data points
        for (int j = 0; j < TRAIN_SIZE / BATCH_SIZE; j++) {
            // starting index
            int starting_idx = j * BATCH_SIZE;
            
            // Select next 10 unvisited datapoints in shuffled array
            for (int n = 0; n < BATCH_SIZE; n++) {
                Entry curr_entry = entries[starting_idx + n];
                // add to bath inputs
                inputs[n * NUM_INPUTS] = curr_entry.x;
                inputs[n * NUM_INPUTS + 1] = curr_entry.y;
                
                // Expected y
                grnd_truth[n * NUM_OUTPUTS] = curr_entry.label;
            }
            Value** in = init_values(inputs, NUM_INPUTS * BATCH_SIZE);
            Value** gt = init_values(grnd_truth, NUM_OUTPUTS * BATCH_SIZE);

            // Train batch
            float batch_loss = train(mlp, in, NUM_INPUTS, gt, lr, BATCH_SIZE, products, out);
            // Add to epoch loss
            epoch_loss += batch_loss;
        }
        // Evaluate Accuracy
        int correct = 0;
        for (int i = 0; i < TEST_SIZE; i++){
            Entry curr_entry = entries[TRAIN_SIZE + i];
            float inputs[NUM_INPUTS] = {curr_entry.x, curr_entry.y};
            Value** curr_in = init_values(inputs, NUM_INPUTS);
            Value** out = mlp_forward(mlp, curr_in, NUM_INPUTS);
            // Calculate Accracy against ground truth
            if (pow((curr_entry.label - out[0]->val),2) <= 0.05) {
                correct++;
            }
        }

        printf("EPOCH: %d LOSS: %f ACCURACY: %.d%%\n", i+1, epoch_loss / TRAIN_SIZE, 100 * correct / TEST_SIZE);
    }

    return EXIT_SUCCESS;
}
