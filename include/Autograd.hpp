#ifndef AUTOGRAD_HPP
#define AUTOGRAD_HPP

#include <memory>

#include "Graph.hpp"
#include "Tensor.hpp"

namespace autograd {

template <typename T>
class Autograd {
    using TensorNode = typename Graph<T>::TensorNode;
    using FunctionNode = typename Graph<T>::FunctionNode;
    using Index = typename Tensor<T>::Index;

    std::unique_ptr<Graph<T>> graph_;

public:
    Autograd();
    Tensor<T>* MakeTensor(const Tensor<T> &t);
    Tensor<T>* MakeTensor(T value, bool requiresGrad = false);
    Tensor<T>* MakeTensor(const Index &shape, const std::vector<T> &value, bool requiresGrad = false);

    Tensor<T>* mult(Tensor<T>* a, Tensor<T>* b);
    Tensor<T>* dot(Tensor<T>* a, Tensor<T>* b);

    void backward(Tensor<T>* tensor);

    Tensor<T>* gradient(Tensor<T>* tensor);
};

template <typename T>
Autograd<T>::Autograd() {
    graph_ = std::make_unique<Graph<T>>();
}

template <typename T>
Tensor<T>* Autograd<T>::MakeTensor(const Tensor<T> &t) {
    Tensor<T>* tensor = new Tensor<T>{t};
    graph_->addTensorNode(tensor, true);
    return tensor;
}

template <typename T>
Tensor<T>* Autograd<T>::MakeTensor(T value, bool requiresGrad) {
    Tensor<T>* tensor = new Tensor<T>{value};
    graph_->addTensorNode(tensor, true, requiresGrad);
    return tensor;
}

template <typename T>
Tensor<T>* Autograd<T>::MakeTensor(const Index &shape, const std::vector<T> &value, bool requiresGrad) {
    Tensor<T>* tensor = new Tensor<T>{shape, value};
    graph_->addTensorNode(tensor, true, requiresGrad);
    return tensor;
}

template <typename T>
Tensor<T>* Autograd<T>::mult(Tensor<T>* a, Tensor<T>* b) {
    if (!Tensor<T>::checkShapes(*a, *b)) {
        throw std::runtime_error("Shapes of tensors must be equal!");
    }
    Mult<T>* multFunction1 = new Mult<T>(b); // for a
    Mult<T>* multFunction2 = new Mult<T>(a); // for b
    Tensor<T>* res = multFunction1->forward(a);
    graph_->addTensorNode(res, false);
    graph_->addFunctionNode(static_cast<Function<T>*>(multFunction1), res, a);
    graph_->addFunctionNode(static_cast<Function<T>*>(multFunction2), res, b);
    return res;
}

template <typename T>
Tensor<T>* Autograd<T>::dot(Tensor<T>* a, Tensor<T>* b) {
    if (a->nDims() < b->nDims()) {
        throw std::runtime_error("nDims of a left tensor must be greater or equal to nDims of a right tensor! a->nDims(): " + std::to_string(a->nDims()) + ". b->nDims(): " + std::to_string(b->nDims()));
    }
    for (size_t i = 0; i < b->nDims(); ++i) {
        if (a->shape()[b->nDims() - i - 1] != b->shape()[b->nDims() - i - 1]) {
            throw std::runtime_error("Shapes of tensors must be equal! a->shape()[i]: " + std::to_string(a->nDims()) + ". b->shape()[i]: " + std::to_string(b->nDims()));
        }
    }
    Dot<T>* dotFunction1 = new Dot<T>(b); // for a
    Dot<T>* dotFunction2 = new Dot<T>(a); // for b
    Tensor<T>* res = dotFunction1->forward(a);
    graph_->addTensorNode(res, false);
    graph_->addFunctionNode(static_cast<Function<T>*>(dotFunction1), res, a);
    graph_->addFunctionNode(static_cast<Function<T>*>(dotFunction2), res, b);
    return res;
}

template <typename T>
void Autograd<T>::backward(Tensor<T>* tensor) {
    graph_->calcGradients(tensor);
}

template <typename T>
Tensor<T>* Autograd<T>::gradient(Tensor<T>* tensor) {
    return graph_->getGradient(tensor);
}

}
#endif // AUTOGRAD_HPP