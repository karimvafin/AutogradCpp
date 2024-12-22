#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <list>

#include <Tensor.hpp>
#include <Function.hpp>

namespace autograd {

template <typename T>
struct Graph {
public:
    struct FunctionNode;

    struct TensorNode {
        Tensor<T>* tensor_;
        std::vector<FunctionNode*> nextFunctions_;
        std::vector<FunctionNode*> prevFunctions_;
        bool isLeaf_;
        bool requiresGrad_;
        Tensor<T>* grad_;

        TensorNode(Tensor<T>* tensor, bool isLeaf, bool requiresGrad) : tensor_(tensor), isLeaf_(isLeaf), requiresGrad_(requiresGrad) {}
        void addNextFunction(FunctionNode* fnode) { nextFunctions_.push_back(fnode); }
        void addPrevFunction(FunctionNode* fnode) { prevFunctions_.push_back(fnode); }
    };

    struct FunctionNode {
        Function<T>* f_;
        TensorNode* nextTensor_;
        TensorNode* prevTensor_;

        FunctionNode(Function<T>* f, TensorNode* nextTensor, TensorNode* prevTensor) : f_(f), nextTensor_(nextTensor), prevTensor_(prevTensor) {}
        Tensor<T>* backward() {
            return f_->backward(prevTensor_->tensor_);
        }
        void setGradientToPrev(Tensor<T>* grad) {
            prevTensor_->grad_ = grad;
        }
    };

    using TensorNodeIt = typename std::list<TensorNode*>::iterator;
    using FunctionNodeIt = typename std::list<FunctionNode*>::iterator;

private:
    std::list<TensorNode*> tensorNodes_;
    std::list<FunctionNode*> functionNodes_;
    std::unordered_map<Tensor<T>*, TensorNodeIt> tensorToNode_;
    std::unordered_map<Function<T>*, FunctionNodeIt> functionToNode_;

    void calcGradientsImpl(TensorNode* tnode, Tensor<T>* grad);

public:
    Graph() = default;
    void addTensorNode(Tensor<T>* tensor, bool isLeaf, bool requiresGrad = false);
    void addFunctionNode(Function<T>* f, Tensor<T>* nextTensor, Tensor<T>* prevTensor);
    void calcGradients(Tensor<T>* tensor);
    Tensor<T>* getGradient(Tensor<T>* tensor);
};

template <typename T>
void Graph<T>::addTensorNode(Tensor<T>* tensor, bool isLeaf, bool requiresGrad) {
    TensorNode* tnode = new TensorNode(tensor, isLeaf, requiresGrad);
    tensorNodes_.push_back(tnode);
    tensorToNode_[tnode->tensor_] = std::prev(tensorNodes_.end());
}

template <typename T>
void Graph<T>::addFunctionNode(Function<T>* f, Tensor<T>* nextTensor, Tensor<T>* prevTensor) {
    FunctionNode* fnode = new FunctionNode(f, *tensorToNode_[nextTensor], *tensorToNode_[prevTensor]);
    (*tensorToNode_[nextTensor])->addPrevFunction(fnode);
    (*tensorToNode_[prevTensor])->addNextFunction(fnode);
    functionNodes_.push_back(fnode);
    functionToNode_[f] = std::prev(functionNodes_.end());
}

template <typename T>
void Graph<T>::calcGradients(Tensor<T>* tensor) {
    TensorNode* tnode = *tensorToNode_[tensor];
    Tensor<T>* grad = new Tensor<T>{*tensor};
    for (size_t i = 0; i < grad->value().size(); ++i) {
        grad->value()[i] = 1;
    }
    tnode->grad_ = grad;
    calcGradientsImpl(tnode, grad);
}

template <typename T>
void Graph<T>::calcGradientsImpl(TensorNode* tnode, Tensor<T>* grad) {
    if (tnode->isLeaf_) return;
    std::vector<FunctionNode*>& fnodes = tnode->prevFunctions_;
    for (size_t i = 0; i < fnodes.size(); ++i) {
        Tensor<T>* newGrad = fnodes[i]->backward();
        newGrad->value()[0] *= grad->value()[0];
        fnodes[i]->setGradientToPrev(newGrad);
        calcGradientsImpl(fnodes[i]->prevTensor_, newGrad);
    }
}
    
template <typename T>
Tensor<T>* Graph<T>::getGradient(Tensor<T>* tensor) {
    return (*tensorToNode_[tensor])->grad_;
}

} // namespace autograd


#endif // GRAPH_HPP