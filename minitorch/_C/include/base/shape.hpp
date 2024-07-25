#ifndef SHAPE_HPP
#define SHAPE_HPP

#include<utility>
#include<vector>
#include<initializer_list>
#include<algorithm>
#include<cstddef>
#include<numeric>
#include<functional>

#include"type_extension.hpp"
#include"base/slice.hpp"


namespace minitorch {
    class Shape {
        std::shared_ptr<Shape> front = nullptr;
        std::vector<Slice> reference_slices = std::vector<Slice>();
        size_t _get_ptr_index(std::initializer_list<size_t> index) const;
        size_t _get_ptr_index(const std::vector<size_t> &index) const;
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
        void set_reference(const std::shared_ptr<Shape> &reference, const std::vector<Slice> &slices);
        size_t get_ptr_index(std::initializer_list<size_t> index) const;
        size_t get_ptr_index(const std::vector<size_t> &index) const;
        std::vector<size_t> get_slice_index_from_ptr_index(size_t ptr_index) const;
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
            this->reference_slices.push_back(Slice());
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

    inline void Shape::set_reference(
        const std::shared_ptr<Shape> &reference,
        const std::vector<Slice> &slices
    ) {
        if (reference != nullptr) {
            this->front = reference;
            auto full_slice = std::vector<Slice>();
            const size_t ndim = reference.get()->ndim();
            const size_t slice_dim = slices.size();
            for (size_t idx = 0; idx < ndim; idx++) {
                if (idx >= slice_dim) {
                    full_slice.push_back(Slice(size_t(0), reference.get()->get_item(idx), slices[idx].get_step()));
                } else {
                    IndexType slice_start_index = slices[idx].get_start();
                    IndexType slice_end_index = slices[idx].get_end();
                    size_t start_index = 0;
                    size_t end_index = 0;
                    if (std::holds_alternative<py::none>(slice_start_index)) {
                        start_index = this->get_item(idx);
                    } else {
                        start_index = std::get<size_t>(slice_start_index);
                    }
                    if (std::holds_alternative<py::none>(slice_end_index)) {
                        end_index = this->get_item(idx);
                    } else {
                        end_index = std::get<size_t>(slice_end_index);
                    }
                    full_slice.push_back(Slice(start_index, end_index, slices[idx].get_step()));
                }
            }
            this->reference_slices = full_slice;
        } else {
            WARNING("函数minitorch::Shape::set_reference警告: 接收了一个nullptr类型的reference参数!");
        }
    }

    inline size_t Shape::get_ptr_index(const std::vector<size_t> &index) const {
        std::shared_ptr<Shape> cur_shape_ptr = this->front;
        auto cur_index = index;
        std::cout << "Start get_ptr_index!" << std::endl;
        while (cur_shape_ptr != nullptr) {
            auto cur_shape = cur_shape_ptr.get();
            auto cur_slice = cur_shape->reference_slices;
            auto cur_full_index = std::vector<size_t>();
            std::cout << cur_shape->to_string() << std::endl;
            if (cur_index.size() < cur_shape->ndim()) {
                for (size_t idx = 0; idx < cur_shape->ndim(); idx++) {
                    size_t start;
                    size_t end;
                    if (std::holds_alternative<py::none>(cur_slice[idx].get_start()))
                        start = 0;
                    else
                        start = std::get<size_t>(cur_slice[idx].get_start());
                    if (std::holds_alternative<py::none>(cur_slice[idx].get_end()))
                        end = cur_shape->get_item(idx);
                    else
                        end = std::get<size_t>(cur_slice[idx].get_end());
                    if (end - start == 0)
                        cur_index.insert(cur_index.begin() + idx, 0);
                }
            }
            for (size_t idx = 0; idx < cur_shape->ndim(); idx++) {
                IndexType start_variant = cur_slice[idx].get_start();
                size_t start;
                if (std::holds_alternative<py::none>(start_variant)) {
                    start = 0;
                } else {
                    start = std::get<size_t>(start_variant);
                }
                IndexType end_variant = cur_slice[idx].get_start();
                cur_full_index.push_back(start + cur_index[idx] * cur_slice[idx].get_step());
            }
            cur_index = cur_full_index;
            cur_shape_ptr = cur_shape->front;
        }
        std::cout << "End get_ptr_index!" << std::endl;
        if (cur_shape_ptr == nullptr)
            return this->_get_ptr_index(cur_index);
        return cur_shape_ptr->_get_ptr_index(cur_index);
    }

    inline size_t Shape::_get_ptr_index(const std::initializer_list<size_t> index) const {
        CHECK(index.size() == this->ndim());
        size_t ptr_index = 0;
        for (int idx = 0; idx < index.size(); idx++) {
            size_t cur_idx = index.begin()[idx];
            ptr_index += cur_idx * this->stride[idx];
        }

        return ptr_index;
    }

    inline size_t Shape::_get_ptr_index(const std::vector<size_t> &index) const {
        CHECK(index.size() == this->ndim());
        size_t ptr_index = 0;
        for (size_t idx = 0; idx < index.size(); idx++) {
            size_t cur_idx = index.begin()[idx];
            ptr_index += cur_idx * this->stride[idx];
        }
        return ptr_index;
    }

    inline std::vector<size_t> Shape::get_slice_index_from_ptr_index(size_t ptr_index) const {
        std::cout << "get_slice_index_from_ptr_index start!" << std::endl;
        std::vector<size_t> index(this->ndim(), 0);
        for (size_t i = this->ndim() - 1; i > 0; i--) {
            index[i - 1] = ptr_index / this->stride[i - 1];
            ptr_index %= this->stride[i - 1];
        }
        // 最后一个维度的索引直接赋值
        index.front() = ptr_index;
        std::reverse(index.begin(), index.end());

        return index;
    }
}

#endif
