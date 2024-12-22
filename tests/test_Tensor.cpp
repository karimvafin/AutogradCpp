#include <iostream>

#include "Tensor.hpp"

int main() {
    /** Tensor construction */
    std::cout << "Tensor construction" << std::endl;
    autograd::Tensor<float> t0(5.0f);
    autograd::Tensor<float> t1({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    std::cout << t0 << std::endl;
    std::cout << t1 << std::endl;

    /** Slicing */
    std::cout << "Slicing" << std::endl;
    autograd::Tensor<float> t2 = t1.at({0});
    autograd::Tensor<float> t3 = t1.at({1, 1});

    std::cout << t2 << std::endl;
    assert(t2.at({0}).item() == 1.0f);
    assert(t2.at({1}).item() == 2.0f);

    std::cout << t3 << std::endl;
    assert(t3.item() == 4.0f);

    /** operator+, operator+= */
    autograd::Tensor<float> t4({3}, {5.0f, 6.0f, 1.5f});
    autograd::Tensor<float> t5({3}, {5.2f, 6.5f, 1.9f});
    autograd::Tensor<float> t6 = t4 + t5;
    assert(t6.at({0}).item() == 10.2f);
    assert(t6.at({1}).item() == 12.5f);
    assert(t6.at({2}).item() == 3.4f);

    autograd::Tensor<float> t7({3}, {1.0f, 2.0f, 3.0f});
    t6 += t7;
    assert(t6.at({0}).item() == 11.2f);
    assert(t6.at({1}).item() == 14.5f);
    assert(t6.at({2}).item() == 6.4f);

    /** Zero / empty as */
    autograd::Tensor<float> t8 = autograd::Tensor<float>::zeroAs(t1);
    autograd::Tensor<float> t9 = autograd::Tensor<float>::emptyAs(t1);
    assert(t8.at({0, 0}).item() == 0.0f);
    assert(t8.at({0, 1}).item() == 0.0f);
    assert(t8.at({1, 0}).item() == 0.0f);
    assert(t8.at({1, 1}).item() == 0.0f);
}