/*
 * Demo Nueral Network Lib with MLP
 */
#include<time.h>

#include "nn.h"
#include "engine.h"

int main() {
    // Seed the random number generator
    srand(time(NULL));

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
