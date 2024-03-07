#include <stdio.h>
#include <stdlib.h>

#include "engine.h"

int main(int argc, char** argv) {
    // Print GPU Info
    printCudaInfo();

    // Test add and subtract
    Value* x = init_value(1.0);
    Value* y = init_value(1.5);

    Value* sum = add(x,y);

    Value* t = init_value(3.0);

    // Z = (X + Y) - T
    Value* z = sub(sum, t);

    backward(z,1);
    print_expression(z);
  
    print_value(z);
    print_value(t);
    print_value(x);
    print_value(y);

    return EXIT_SUCCESS;
}
