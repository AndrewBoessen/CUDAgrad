#include <stdio.h>
#include <stdlib.h>

#include "engine.h"

int main(int argc, char** argv) {
    // Test add and subtrac
    Value* x = init_value(1.0);
    Value* y = init_value(1.5);

    Value* sum = add(x,y);

    Value* t = init_value(3.0);

    // Z = (X + Y) - T
    Value* z = sub(sum, t);

    print_children(z);

    return EXIT_SUCCESS;
}