#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <numeric>
#include <stdexcept>
#include <assert.h>
#include <ostream>

namespace autograd {

template <typename T>
class Tensor {
public:
    using Index = std::vector<unsigned int>;

private:

    unsigned int nDims_;
    Index shape_;
    unsigned int totalNumOfElements_;
    std::vector<T> value_;

    size_t calcInternalIndex(const Index &index) const;
    bool checkIndexValidity(const Index &index) const;

public:
    static Tensor<T> zeroAs(const Tensor<T> &t);
    static Tensor<T> emptyAs(const Tensor<T> &t);
    Tensor(T value);
    Tensor(const Index &shape, const std::vector<T> &value);
    ~Tensor() {}
    Tensor<T> at(const Index &index) const;
    T item() const;
    std::vector<T>& value();
    const Index& shape() const;
    unsigned int nDims() const;

    T& operator()(const Index& index);
    Tensor<T>& operator+=(const Tensor<T>& other);
    Tensor<T> operator+(const Tensor<T>& other) const;
    
    static bool checkShapes(const Tensor<T>& first, const Tensor<T>& second);

    template <typename U>
    friend std::ostream& operator<<(std::ostream& os, const Tensor<U>& tensor);
};

template <typename T>
Tensor<T> Tensor<T>::zeroAs(const Tensor<T> &t) {
    return Tensor<T>{t.shape_, std::vector<T>(t.totalNumOfElements_, 0)};
}

template <typename T>
Tensor<T> Tensor<T>::emptyAs(const Tensor<T> &t) {
    return Tensor<T>{t.shape_, std::vector<T>(t.totalNumOfElements_)};
}

template <typename T>
Tensor<T>::Tensor(T value) {
    nDims_ = 0;
    totalNumOfElements_ = 1;
    value_.resize(totalNumOfElements_);
    value_[0] = value;
}

template <typename T>
Tensor<T>::Tensor(const Index &shape, const std::vector<T> &value) {
    totalNumOfElements_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<unsigned int>());
    if (value.size() != totalNumOfElements_) {
        throw std::runtime_error("Number of elements vector must be equal to the accum of shape! value.size() = " 
        + std::to_string(value.size()) + ", totalNumOfElements = " + std::to_string(totalNumOfElements_));
    }
    assert(value.size() == totalNumOfElements_);
    nDims_ = shape.size();
    shape_ = shape;
    value_ = value;
}

template <typename T>
bool Tensor<T>::checkIndexValidity(const Index &index) const {
    if (index.size() != nDims_) {
        return false;
    }
    for (size_t i = 0; i < nDims_; ++i) {
        if (index[i] >= shape_[i]) {
            return false;
        }
    }
    return true;
}

template <typename T>
size_t Tensor<T>::calcInternalIndex(const Index &index) const {
    bool indexIsValid = checkIndexValidity(index);
    if (!indexIsValid) {
        throw std::runtime_error("Index is invalid!");
    }
    size_t res = 0;
    size_t accum = 1;
    for (size_t i = 0; i < nDims_; ++i) {
        res += accum * index[nDims_ - i - 1];
        accum *= shape_[nDims_ - i - 1];
    }
    return res;
}

template <typename T>
Tensor<T> Tensor<T>::at(const Index &index) const {
    size_t indexSize = index.size();
    assert(0 < indexSize && indexSize <= nDims_);

    unsigned int nDims = nDims_ - indexSize;
    Index shape{shape_.begin() + indexSize, shape_.end()};
    assert(shape.size() == nDims);

    unsigned int totalNumOfElements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<unsigned int>());
    /** Calculating internal index */
    size_t internalIndex = 0;
    size_t accum = totalNumOfElements;
    for (size_t i = 0; i < indexSize; ++i) {
        internalIndex += accum * index[indexSize - i - 1];
        accum *= shape_[indexSize - i - 1];
    }
    return Tensor<T>{shape, std::vector<T>{value_.begin() + internalIndex, value_.begin() + internalIndex + totalNumOfElements}};
}

template <typename T>
T Tensor<T>::item() const {
    if (nDims_ > 0) {
        throw std::runtime_error("item() member function is available only for scalars! nDims_: " + std::to_string(nDims_));
    }
    return value_[0];
}

template <typename T>
std::vector<T>& Tensor<T>::value() {
    return value_;
}

template <typename T>
const typename Tensor<T>::Index& Tensor<T>::shape() const {
    return shape_;
}

template <typename T>
unsigned int Tensor<T>::nDims() const {
    return nDims_;
}

template <typename T>
T& Tensor<T>::operator()(const Index &index) {
    return value_[calcInternalIndex(index)];
}

template <typename T>
Tensor<T>& Tensor<T>::operator+=(const Tensor<T>& other) {
    if (!Tensor<T>::checkShapes(*this, other)) {
        throw std::invalid_argument("Shapes of tensors are not equal!");
    }
    for (size_t i = 0; i < totalNumOfElements_; ++i) {
        value_[i] += other.value_[i];
    }
    return *this;
}

template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
    Tensor<T> result = *this;
    result += other; 
    return result; 
}

template <typename T>
bool Tensor<T>::checkShapes(const Tensor<T>& first, const Tensor<T>& second) {
    if (first.nDims() != second.nDims()) {
        return false;
    }
    for (size_t i = 0; i < first.nDims(); ++i) {
        if (first.shape()[i] != second.shape()[i]) {
            return false;
        }
    }
    return true;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
    if (tensor.nDims_ == 0) {
        os << tensor.value_[0];
    } else if (tensor.nDims_ == 2) {
        os << "[";
        for (size_t i = 0; i < tensor.shape_[0]; ++i) {
            for (size_t j = 0; j < tensor.shape_[1]; ++j) {
                if (j == 0 && i != 0) os << " ";
                if (j == 0) os << "[";
                os << tensor.value_[tensor.shape_[1] * i + j];
                if (j != (tensor.shape_[1] - 1)) 
                    os << ", "; 
                else 
                    os << "]";
            }
            if (i != (tensor.shape_[0] - 1)) os << "\n";
        }
        os << "]";
    } else {
        os << "[";
        for (size_t i = 0; i < tensor.totalNumOfElements_; ++i) {
            os << tensor.value_[i];
            if (i != (tensor.totalNumOfElements_ - 1)) os << ", ";
        }
        os << "]";
    }
    return os;
}


} // namespace autograd

#endif // TENSOR_HPP