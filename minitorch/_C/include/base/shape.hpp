#ifndef SHAPE_HPP
#define SHAPE_HPP

#include<utility>
#include<vector>
#include<initializer_list>
#include<algorithm>
#include<cstddef>
#include<numeric>
#include<functional>


namespace minitorch {
    class Shape {
    public:
        std::vector<size_t> shape = std::vector<size_t>();
        std::vector<size_t> stride = std::vector<size_t>();
        Shape();
        Shape(std::initializer_list<size_t> sizes);
        explicit Shape(std::vector<size_t> sizes);
        size_t ndim() const;
        size_t get_item(size_t index) const;
        void set_item(size_t index, size_t item);
        size_t elem_size() const;
        std::string to_string() const;
    };

    inline Shape::Shape() = default;

    inline Shape::Shape(std::initializer_list<size_t> sizes) {
        for (auto size: sizes) {
            this->shape.push_back(size);
        }
        for (int idx = 0; idx < this->ndim(); idx++) {
            if (idx == 0) {
                this->stride.push_back(size_t(1));
            } else {
                this->stride.push_back(
                    this->shape[this->ndim() - idx] * this->stride[idx - 1]
                );
            }
        }
        std::reverse(this->stride.begin(), this->stride.end());
    } 

    inline Shape::Shape(std::vector<size_t> sizes) {
        this->shape = std::move(sizes);
        for (int idx = 0; idx < this->ndim(); idx++) {
            if (idx == 0) {
                this->stride.push_back(size_t(1));
            } else {
                this->stride.push_back(
                    this->shape[this->ndim() - idx] * this->stride[idx - 1]
                );
            }
        }
        std::reverse(this->stride.begin(), this->stride.end());
    }

    inline size_t Shape::ndim() const {
        return this->shape.size();
    }

    inline size_t Shape::get_item(const size_t index) const {
        return this->shape[index];
    }

    inline void Shape::set_item(const size_t index, const size_t item) {
        this->shape[index] = item;
    }

    inline size_t Shape::elem_size() const {
        return std::accumulate(this->shape.begin(), this->shape.end(), 1, std::multiplies<size_t>{});
    }

    inline std::string Shape::to_string() const {
        std::ostringstream os_string;
        os_string << "(";
        for (size_t i = 0; i < this->shape.size(); i++) {
            os_string << this->shape[i];
            if (i < this->shape.size() - 1) {
                os_string << ", ";
            }
        }
        os_string << ")";
        return os_string.str();
    }

}

#endif
