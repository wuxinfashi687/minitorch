#ifndef TENSOR_HPP
#define TENSOR_HPP

#include"shape.hpp"
#include"dtype.hpp"
#include"buffer.hpp"


namespace minitorch {
    class Tensor {
    protected:
        std::shared_ptr<Buffer> buffer = nullptr;
        Shape shape;
        DType dtype = DType(DTypeEnum::KUnKnown);
    public:
        Tensor(std::shared_ptr<MemoryAllocator> allocator, const Shape& shape, DType dtype);
        Tensor(const Tensor&) = delete;
        Tensor& operator=(const Tensor&) = delete;
        Tensor(Tensor&&) = delete;
        Tensor& operator=(Tensor&&) = delete;
        Shape get_shape() const;
        DType get_dtype() const;
        void* ptr() const;
        template<typename T>
        T* ptr();
    };

    inline Tensor::Tensor(std::shared_ptr<MemoryAllocator> allocator, const Shape& shape, const DType dtype) {
        this->buffer = std::make_shared<Buffer>(size_t(1), allocator, nullptr);
        this->shape = shape;
        this->dtype = dtype;
    }

    inline Shape Tensor::get_shape() const {
        return this->shape;
    }

    inline DType Tensor::get_dtype() const {
        return this->dtype;
    }

    inline void* Tensor::ptr() const {
        return this->buffer.get()->ptr();
    }

    template <typename T>
    T* Tensor::ptr() {
        void* points = this->ptr();
        return reinterpret_cast<T>(points);
    }
}

#endif
