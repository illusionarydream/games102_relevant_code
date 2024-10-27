#include <iostream>
#include <fitting.hpp>

int main() {
    vector_d x = {0, 1, 2, 3, 4, 5};
    vector_d y = {0, 10, 0, 10, 0, 10};

    Fitting fitting;
    fitting.polygon_fitting(x, y, 3);
    fitting.draw(0, 5.1);

    fitting.polygon_fitting(x, y, 5);
    fitting.draw(0, 5.1);

    return 0;
}