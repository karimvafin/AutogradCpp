#include <iostream>

#include "Autograd.hpp"

using namespace autograd;

int main() {
    Autograd<float> ag;
    Tensor<float>* a = ag.MakeTensor(3, true);
    Tensor<float>* b = ag.MakeTensor(4, true);

    Tensor<float>* c = ag.dot(a, b);

    Tensor<float>* d = ag.MakeTensor(5, false);
    Tensor<float>* e = ag.dot(c, d);

    std::cout << "a: " << *a << std::endl;
    std::cout << "b: " << *b << std::endl;
    std::cout << "c = dot(a, b): " << *c << std::endl;
    std::cout << "d: " << *d << std::endl;
    std::cout << "e = dot(c, d): " << *e << std::endl;

    ag.backward(e);

    std::cout << "grad(a): " << *(ag.gradient(a)) << std::endl;
    std::cout << "grad(b): " << *(ag.gradient(b)) << std::endl;
}