#ifndef FUNCTION_HPP
#define FUNCTION_HPP

#include "Tensor.hpp"

namespace autograd {

template <typename T>
class Function {
public:
    virtual Tensor<T>* forward(Tensor<T>* x) = 0;
    virtual Tensor<T>* backward(Tensor<T>* y) = 0;
};

template <typename T>
class Exp : public Function<T> {
public:
    Tensor<T>* forward(Tensor<T>* x) override {
        Tensor<T>* res = new Tensor<T>{*x};
        std::vector<T>& value = res->value();
        for (size_t i = 0; i < value.size(); ++i) {
            value[i] = std::exp(x->value()[i]);
        }
        return res;
    }

    Tensor<T>* backward(Tensor<T>* x) override {
        Tensor<T>* res = new Tensor<T>{*x};
        std::vector<T>& value = res->value();
        for (size_t i = 0; i < value.size(); ++i) {
            value[i] = std::exp(x->value()[i]);
        }
        return res;
    }
};

template <typename T>
class Mult : public Function<T> {
    Tensor<T>* multiplier_;

public:
    Mult(Tensor<T>* multiplier) : multiplier_(multiplier) {}

    Tensor<T>* forward(Tensor<T>* x) override {
        Tensor<T>* res = new Tensor<T>{*x};
        std::vector<T>& value = res->value();
        for (size_t i = 0; i < value.size(); ++i) {
            value[i] *= multiplier_->value()[i];
        }
        return res;
    }

    Tensor<T>* backward(Tensor<T>* x) override {
        Tensor<T>* res = new Tensor<T>{*x};
        std::vector<T>& value = res->value();
        for (size_t i = 0; i < value.size(); ++i) {
            value[i] = multiplier_->value()[i];
        }
        return res;
    }
};

template <typename T>
class Dot : public Function<T> {
    Tensor<T>* multiplier_;

public:
    Dot(Tensor<T>* multiplier) : multiplier_(multiplier) {}

    Tensor<T>* forward(Tensor<T>* x) override {
        T res = x->value()[0] * multiplier_->value()[0];
        std::vector<T>& value = x->value();
        for (size_t i = 1; i < value.size(); ++i) {
            res += value[i] * multiplier_->value()[i];
        }
        return new Tensor<T>{res};
    }

    Tensor<T>* backward(Tensor<T>* x) override {
        Tensor<T>* res = new Tensor<T>{*x};
        std::vector<T>& value = res->value();
        for (size_t i = 0; i < value.size(); ++i) {
            value[i] = multiplier_->value()[i];
        }
        return res;
    }
};


} // namespace autograd

#endif // FUNCTION_HPP