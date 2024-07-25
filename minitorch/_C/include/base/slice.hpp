//
// Created by JUMPWORK-SERVER-01 on 2024/7/23.
//

#ifndef SLICE_HPP
#define SLICE_HPP

#include<pybind11/pybind11.h>
#include<variant>

namespace py = pybind11;

namespace minitorch {
    using NewDim = py::none;
    using IndexType = std::variant<size_t, NewDim>;


    class Slice {
        IndexType start = py::none();
        IndexType end = py::none();
        size_t step = 1;
    public:
        Slice(const IndexType &start, const IndexType &end, const size_t step = 1) {
            CHECK(start <= end);
            this->start = start;
            this->end = end;
            this->step = step;
        }
        Slice() = default;
        ~Slice() = default;
        IndexType get_start() const;
        IndexType get_end() const;
        size_t get_step() const;
        void set_start(const IndexType& start);
        void set_end(const IndexType& end);
        void set_step(size_t step);
    };

    inline void Slice::set_start(const IndexType& start)
    {
        this->start = start;
    }

    inline void Slice::set_end(const IndexType& end) {
        this->end = end;
    }

    inline void Slice::set_step(const size_t step)
    {
        this->step = step;
    }

    inline IndexType Slice::get_start() const {
        return this->start;
    }

    inline IndexType Slice::get_end() const {
        return this->end;
    }

    inline size_t Slice::get_step() const {
        return this->step;
    }
}


#endif //SLICE_HPP
