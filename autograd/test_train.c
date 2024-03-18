/*
 * Train MLP
 */
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#include "nn.h"
#include "engine.h"
#include "data.h"

int main() {
    // Seed the random number generator
    srand(time(NULL));

    int num_in = 1;
    int num_out = 1;

    int size[] = {num_in, 5, num_out};
    int nlayers = 3;

    MLP* mlp = init_mlp(size, nlayers);

    int batch_size = 5;

    float inputs[5] = {1, 2, 3, 4, 5};

    Value** in_vals = init_values(inputs, 5);

    float gt_num[5] = {2,4,6,8, 10};

    Value** gt_vals = init_values(gt_num, 5);

    train(mlp, in_vals, num_in, gt_vals, 1, batch_size);

}