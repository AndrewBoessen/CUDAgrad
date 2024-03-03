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

    print_expression(z);
    
    // Add 4 Values together
    Value* x1 = init_value(1);
    Value* x2 = init_value(2);
    Value* x3 = init_value(3);
    Value* x4 = init_value(4);

    Value* s1 = add(x1, x2);
    Value* s2 = add(x3, x4);

    Value* sum_all = add(s1, s2);

    print_expression(sum_all);

    // Multiply and Divide
    Value* a = init_value(15);
    Value* b = init_value(5);
    
    Value* prod = mul(a, b);

    Value* quotient = divide(prod, init_value(5));

    print_expression(quotient);

    return EXIT_SUCCESS;
}