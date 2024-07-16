#ifndef SHAPE_HPP
#define SHAPE_HPP

#include <utility>
#include<vector>
#include<initializer_list>
#include <cstdint>
#include <cstddef>


namespace minitorch {
    class Shape {
    public:
        std::vector<size_t> shape = std::vector<size_t>();
        std::vector<size_t> stride = std::vector<size_t>();
        Shape();
        Shape(std::initializer_list<size_t> sizes);
        explicit Shape(std::vector<size_t> sizes);
        size_t ndim() const;
        size_t get_item(size_t index);
        void set_item(size_t index, size_t item);
    };

    inline Shape::Shape() = default;

    inline Shape::Shape(std::initializer_list<size_t> sizes) {
        for (auto size: sizes) {
            this->shape.push_back(size);
        }
        for (int idx = 0; idx <= this->ndim(); idx ++) {
            if (idx == 0) {
                this->stride.push_back(size_t(1));
            } else {
                this->stride.push_back(
                    this->shape[this->ndim() - idx - 1] * this->stride[idx - 1]
                );
            }
        }
    } 

    inline Shape::Shape(std::vector<size_t> sizes) {
        this->shape = std::move(sizes);
        for (int idx = 0; idx <= this->ndim(); idx ++) {
            if (idx == 0) {
                this->stride.push_back(size_t(1));
            } else {
                this->stride.push_back(
                    this->shape[this->ndim() - idx - 1] * this->stride[idx - 1]
                );
            }
        }
    }

    inline size_t Shape::ndim() const {
        return this->shape.size();
    }

    inline size_t Shape::get_item(const size_t index) {
        return this->shape[index];
    }

    inline void Shape::set_item(const size_t index, const size_t item) {
        this->shape[index] = item;
    }
}

#endif
