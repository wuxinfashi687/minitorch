#ifndef TENSOR_HPP
#define TENSOR_HPP

#include"shape.hpp"
#include"dtype.hpp"
#include"buffer.hpp"
#include"slice.hpp"


namespace minitorch {
    class Tensor {
    protected:
        std::shared_ptr<Buffer> buffer = nullptr;
        Shape shape;
        DType dtype = DType(DTypeEnum::KUnKnown);
    public:
        Tensor(std::shared_ptr<MemoryAllocator> allocator, Shape shape, DType dtype);
        // Tensor(const Tensor&) = delete;
        // Tensor& operator=(const Tensor&) = delete;
        // Tensor(Tensor&&) = delete;
        // Tensor& operator=(Tensor&&) = delete;
        Shape get_shape() const;
        DType get_dtype() const;
        void* ptr() const;
        template<typename T>
        T* ptr() const;
        template<typename T>
        T ptr(size_t index);
        Tensor get_item(std::initializer_list<Slice> index);
        Tensor get_item(std::vector<Slice> index);
        Tensor get_item(Slice* index);
        Tensor operator[](std::initializer_list<Slice> index);
        std::string to_string() const;
    };

    inline Tensor::Tensor(std::shared_ptr<MemoryAllocator> allocator, Shape shape, const DType dtype) {
        this->buffer = std::make_shared<Buffer>(size_t(shape.elem_size() * dtype.byte_size()), allocator, nullptr);
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
    T* Tensor::ptr() const {
        void* points = this->ptr();
        return static_cast<T*>(points);
    }

    template <typename T>
    T Tensor::ptr(size_t index) {
        CHECK(this->ptr() != nullptr && this->buffer != nullptr);
        CHECK(index < this->shape.elem_size());
        void* points = this->ptr();
        T* cast_ptr = static_cast<T*>(points);
        return cast_ptr[index];
    }

    inline Tensor Tensor::get_item(std::vector<Slice> index) {
        CHECK(index.size() <= this->shape.ndim());
        auto shape_vector = std::vector<size_t>();
        for (size_t idx = 0; idx < this->shape.ndim(); idx++) {
            size_t shape_item;
            if (idx >= index.size()) {
                shape_item = this->shape.get_item(idx);
            } else {
                Slice cur_slice = index[idx];
                auto cur_start = cur_slice.get_start();
                auto cur_end = cur_slice.get_end();
                if (std::holds_alternative<py::none>(cur_start)) {
                    index[idx].set_start(0);
                }
                if (std::holds_alternative<py::none>(cur_end)) {
                    index[idx].set_end(this->shape.get_item(idx));
                }
                size_t start = std::get<size_t>(cur_start);
                size_t end = std::get<size_t>(cur_end);
                CHECK(cur_slice.get_step() != 0);
                shape_item = std::ceil(static_cast<float>((end - start) / cur_slice.get_step()));
            }
            if (shape_item != 0) {

            }
        }
    }

    inline std::string Tensor::to_string() const {
        std::ostringstream os_string;
        os_string << "{" << std::endl;
        os_string << "\t" << "Shape: " << this->shape.to_string() << "," << std::endl;
        os_string << "\t" << "Data: (";
        for (int idx = 0; idx < this->shape.elem_size() - 1; idx++) {
            os_string << this->ptr<float_t>()[idx] << ", ";
        }
        os_string << this->ptr<float_t>()[this->shape.elem_size() - 1] << ")" << std::endl;
        os_string << "}" << std::endl;
        return os_string.str();
    }


    inline Tensor zeros(Shape shape, DType dtype, DeviceEnum device_enum) {
        if (device_enum == DeviceEnum::KCpu) {
            Tensor tensor(HostAllocatorFactory::get_instance(), shape, dtype);
            HostAllocatorFactory::get_instance()->memset_zero(tensor.ptr(), size_t(shape.elem_size() * dtype.byte_size()));
            return tensor;
        }
        if (device_enum == DeviceEnum::KCuda) {
            Tensor tensor(CudaAllocatorFactory::get_instance(), shape, dtype);
            CudaAllocatorFactory::get_instance()->memset_zero(tensor.ptr(), size_t(shape.elem_size() * dtype.byte_size()));
            return tensor;
        }
        FATAL("未知的设备类型: " << static_cast<int>(device_enum) << "!");
    }
}

#endif
