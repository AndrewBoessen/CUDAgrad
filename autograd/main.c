#include <stdio.h>
#include <stdlib.h>

#include "engine.h"

int main(int argc, char** argv) {
    // Print GPU Info
    printCudaInfo();

    Value* x = init_value(-2);
    Value* y = init_value(5);
    Value* z = init_value(-4);

    Value* q = add(x,y);
    Value* f = mul(q,z);

    backward(f);

    print_expression(f);
    print_value(x);
    print_value(y);
    print_value(z);

    return EXIT_SUCCESS;
}
