/*
 * Demo Nueral Network Lib with MLP
 */
#include<time.h>
#include<stdio.h>

#include "nn.h"
#include "engine.h"

int main() {
    // Seed the random number generator
    srand(time(NULL));

    int n_inputs = 2;
    int n_outputs = 2;

    int sizes[] = {n_inputs, 5, n_outputs};
    int nlayers = sizeof(sizes) / sizeof(int);

    MLP* mlp = init_mlp(sizes, nlayers);
    //show_params(mlp);

    // Allocate inputs
    Value** in;
    allocValueArr(&in, n_inputs);
    // Set inputs
    in[0] = init_value(1.0);
    in[1] = init_value(1.0);

    Value** out = mlp_forward(mlp, in, n_inputs);
    
    float expected[2] = {1, 0};
    Value** gt = init_values(expected, 2);

    Value* loss = mse_loss(out, gt, 2);

    printf("Outputs: \n");
    print_value(out[0]);
    print_value(out[1]);

    backward(loss);

    printf("LOSS:\n");
    print_value(loss);

    //print_expression(out[0]);
    show_params(mlp);
}
