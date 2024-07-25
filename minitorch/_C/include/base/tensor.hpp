#ifndef TENSOR_HPP
#define TENSOR_HPP

#include<memory>

#include"shape.hpp"
#include"dtype.hpp"
#include"buffer.hpp"
#include"slice.hpp"

#ifndef MINITORCH_MAX_NDIM
    #define MINITORCH_MAX_NDIM 10
#endif
#ifndef MINITORCH_PRINT_NUM
    #define MINITORCH_PRINT_NUM 100
#endif


namespace minitorch {
    class Tensor {
    protected:
        std::shared_ptr<Buffer> buffer = nullptr;
        Shape shape;
        DType dtype = DType(DTypeEnum::KUnKnown);
    public:
        Tensor(std::shared_ptr<MemoryAllocator> allocator, Shape shape, DType dtype);
        Tensor(std::shared_ptr<Buffer> buffer, Shape shape, DType dtype);
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
        DeviceEnum get_device() const;
    };

    inline Tensor::Tensor(std::shared_ptr<MemoryAllocator> allocator, Shape shape, const DType dtype) {
        CHECK(shape.ndim() <= MINITORCH_MAX_NDIM);
        this->buffer = std::make_shared<Buffer>(size_t(shape.elem_size() * dtype.byte_size()), allocator, nullptr);
        this->shape = shape;
        this->dtype = dtype;
    }

    inline Tensor::Tensor(std::shared_ptr<Buffer> buffer, Shape shape, DType dtype) {
        CHECK(shape.ndim() <= MINITORCH_MAX_NDIM);
        this->buffer = buffer;
        this->shape = shape;
        this->dtype = dtype;
    }

    inline Shape Tensor::get_shape() const {
        return this->shape;
    }

    inline DeviceEnum Tensor::get_device() const {
        return this->buffer.get()->device_type();
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
                    index[idx].set_start(size_t(0));
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
                shape_vector.push_back(shape_item);
            }
        }
        auto new_shape = Shape(shape_vector);
        auto new_dtype = DType(this->dtype.type);
        std::shared_ptr<Buffer> new_buffer;
        switch (this->get_device()) {
            case DeviceEnum::KCpu: {
                new_buffer = std::make_shared<Buffer>(
                    this->buffer.get()->byte_size(),
                    HostAllocatorFactory::get_instance(),
                    this->ptr(),
                    true
                );
                break;
            }
            case DeviceEnum::KCuda: {
                new_buffer = std::make_shared<Buffer>(
                    this->buffer.get()->byte_size(),
                    CudaAllocatorFactory::get_instance(),
                    this->ptr(),
                    true
                );
                break;
            }
            default:
                FATAL("不支持的Device类型: " << static_cast<int>(this->get_device()) << "!");
        }
        auto new_tensor = Tensor(new_buffer, new_shape, new_dtype);
        new_tensor.shape.set_reference(std::make_shared<Shape>(this->shape), index);
        return new_tensor;
    }

    inline std::string Tensor::to_string() const {
        std::ostringstream os_string;
        os_string << "{" << std::endl;
        os_string << "\t" << "Shape: " << this->shape.to_string() << "," << std::endl;
        os_string << "\t" << "Data: (";
        switch (this->get_dtype().type)
        {
            case DTypeEnum::KBool: {
                for (int idx = 0; idx < this->shape.elem_size() - 1; idx++) {
                    const size_t ptr_index = this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(idx));
                    std::cout << "ptr_index: " << ptr_index << std::endl;
                    os_string << this->ptr<bool>()[ptr_index] << ", ";
                }
                os_string << this->ptr<bool>()[this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(this->shape.elem_size() - 1))] << ")" << std::endl;
                break;
            }
            case DTypeEnum::KFloat32: {
                for (int idx = 0; idx < this->shape.elem_size() - 1; idx++) {
                    const size_t ptr_index = this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(idx));
                    std::cout << "ptr_index: " << ptr_index << std::endl;
                    os_string << this->ptr<float_t>()[ptr_index] << ", ";
                }
                os_string << this->ptr<float_t>()[this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(this->shape.elem_size() - 1))] << ")" << std::endl;
                break;
            }
            case DTypeEnum::KFloat64: {
                for (int idx = 0; idx < this->shape.elem_size() - 1; idx++) {
                    const size_t ptr_index = this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(idx));
                    std::cout << "ptr_index: " << ptr_index << std::endl;
                    os_string << this->ptr<double_t>()[ptr_index] << ", ";
                }
                os_string << this->ptr<double_t>()[this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(this->shape.elem_size() - 1))] << ")" << std::endl;
                break;
            }
            case DTypeEnum::KInt8: {
                for (int idx = 0; idx < this->shape.elem_size() - 1; idx++) {
                    const size_t ptr_index = this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(idx));
                    std::cout << "ptr_index: " << ptr_index << std::endl;
                    os_string << this->ptr<int8_t>()[ptr_index] << ", ";
                }
                os_string << this->ptr<int8_t>()[this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(this->shape.elem_size() - 1))] << ")" << std::endl;
                break;
            }
            case DTypeEnum::KInt16: {
                for (int idx = 0; idx < this->shape.elem_size() - 1; idx++) {
                    const size_t ptr_index = this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(idx));
                    std::cout << "ptr_index: " << ptr_index << std::endl;
                    os_string << this->ptr<int16_t>()[ptr_index] << ", ";
                }
                os_string << this->ptr<int16_t>()[this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(this->shape.elem_size() - 1))] << ")" << std::endl;
                break;
            }
            case DTypeEnum::KInt32: {
                for (int idx = 0; idx < this->shape.elem_size() - 1; idx++) {
                    const size_t ptr_index = this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(idx));
                    std::cout << "ptr_index: " << ptr_index << std::endl;
                    os_string << this->ptr<int32_t>()[ptr_index] << ", ";
                }
                os_string << this->ptr<int32_t>()[this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(this->shape.elem_size() - 1))] << ")" << std::endl;
                break;
            }
            case DTypeEnum::KInt64: {
                for (int idx = 0; idx < this->shape.elem_size() - 1; idx++) {
                    const size_t ptr_index = this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(idx));
                    std::cout << "ptr_index: " << ptr_index << std::endl;
                    os_string << this->ptr<int64_t>()[ptr_index] << ", ";
                }
                os_string << this->ptr<int64_t>()[this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(this->shape.elem_size() - 1))] << ")" << std::endl;
                break;
            }
            case DTypeEnum::KUInt8: {
                for (int idx = 0; idx < this->shape.elem_size() - 1; idx++) {
                    const size_t ptr_index = this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(idx));
                    std::cout << "ptr_index: " << ptr_index << std::endl;
                    os_string << this->ptr<uint8_t>()[ptr_index] << ", ";
                }
                os_string << this->ptr<uint8_t>()[this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(this->shape.elem_size() - 1))] << ")" << std::endl;
                break;
            }
            case DTypeEnum::KUInt16: {
                for (int idx = 0; idx < this->shape.elem_size() - 1; idx++) {
                    const size_t ptr_index = this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(idx));
                    std::cout << "ptr_index: " << ptr_index << std::endl;
                    os_string << this->ptr<uint16_t>()[ptr_index] << ", ";
                }
                os_string << this->ptr<uint16_t>()[this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(this->shape.elem_size() - 1))] << ")" << std::endl;
                break;
            }
            case DTypeEnum::KUInt32: {
                for (int idx = 0; idx < this->shape.elem_size() - 1; idx++) {
                    const size_t ptr_index = this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(idx));
                    std::cout << "ptr_index: " << ptr_index << std::endl;
                    os_string << this->ptr<uint32_t>()[ptr_index] << ", ";
                }
                os_string << this->ptr<uint32_t>()[this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(this->shape.elem_size() - 1))] << ")" << std::endl;
                break;
            }
            case DTypeEnum::KUInt64: {
                for (int idx = 0; idx < this->shape.elem_size() - 1; idx++) {
                    const size_t ptr_index = this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(idx));
                    std::cout << "ptr_index: " << ptr_index << std::endl;
                    os_string << this->ptr<uint64_t>()[ptr_index] << ", ";
                }
                os_string << this->ptr<uint64_t>()[this->shape.get_ptr_index(this->shape.get_slice_index_from_ptr_index(this->shape.elem_size() - 1))] << ")" << std::endl;
                break;
            }
            default:
                FATAL("未知的DTypeEnum类型: " << static_cast<int>(this->get_dtype().type) << "!");
        }
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
