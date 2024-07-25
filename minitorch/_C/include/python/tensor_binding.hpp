#ifndef TENSOR_BINDING_HPP
#define TENSOR_BINDING_HPP

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include"base.h"

namespace py = pybind11;


namespace minitorch {
    namespace python {
        class TensorBinding {
            std::shared_ptr<Tensor> tensor_;
        public:
            TensorBinding(py::buffer memory, Shape shape, DType dtype, DeviceEnum device_enum);
            TensorBinding(Tensor tensor);
            Shape get_shape() const;
            DType get_dtype() const;
            py::buffer_info get_buffer() const;
            ~TensorBinding() = default;
            std::string to_string() const;
            GenericRawDType flatten_get(const py::int_ &index) const;
            TensorBinding get_item(const std::vector<Slice> &slices) const;
        };

        inline TensorBinding::TensorBinding(py::buffer memory, Shape shape, DType dtype, DeviceEnum device_enum) {
            py::buffer_info memory_info = memory.request();
            switch (device_enum) {
                case DeviceEnum::KCpu: {
                    this->tensor_ = std::make_shared<Tensor>(HostAllocatorFactory::get_instance(), shape, dtype);
                    break;
                }
                case DeviceEnum::KCuda: {
                    this->tensor_ = std::make_shared<Tensor>(CudaAllocatorFactory::get_instance(), shape, dtype);
                    break;
                }
                case DeviceEnum::KUnknown: {
                    throw std::runtime_error("不可以使用DeviceEnum::KUnknown作为创建Tensor的参数!");
                }
                default:
                    throw std::runtime_error("非法的Device对象!");
            }
        }

        inline TensorBinding::TensorBinding(Tensor tensor) {
            this->tensor_ = std::make_shared<Tensor>(tensor);
        }

        inline Shape TensorBinding::get_shape() const {
            return this->tensor_.get()->get_shape();
        }

        inline DType TensorBinding::get_dtype() const {
            return this->tensor_.get()->get_dtype();
        }

        inline py::buffer_info TensorBinding::get_buffer() const {
            const DType dtype = this->tensor_.get()->get_dtype();
            const Shape shape = this->tensor_.get()->get_shape();
            void* points = this->tensor_.get()->ptr();
            switch (dtype.type) {
                case DTypeEnum::KInt32:
                    return py::buffer_info(
                        points,
                        dtype.byte_size(),
                        py::format_descriptor<int32_t>::format(),
                        shape.ndim(),
                        shape.shape,
                        shape.stride
                    );
                case DTypeEnum::KInt64:
                    return py::buffer_info(
                        points,
                        dtype.byte_size(),
                        py::format_descriptor<int64_t>::format(),
                        shape.ndim(),
                        shape.shape,
                        shape.stride
                    );
                case DTypeEnum::KInt16:
                    return py::buffer_info(
                        points,
                        dtype.byte_size(),
                        py::format_descriptor<int16_t>::format(),
                        shape.ndim(),
                        shape.shape,
                        shape.stride
                    );
                case DTypeEnum::KInt8:
                    return py::buffer_info(
                        points,
                        dtype.byte_size(),
                        py::format_descriptor<int8_t>::format(),
                        shape.ndim(),
                        shape.shape,
                        shape.stride
                    );
                case DTypeEnum::KUInt32:
                    return py::buffer_info(
                        points,
                        dtype.byte_size(),
                        py::format_descriptor<uint32_t>::format(),
                        shape.ndim(),
                        shape.shape,
                        shape.stride
                    );
                case DTypeEnum::KUInt64:
                    return py::buffer_info(
                        points,
                        dtype.byte_size(),
                        py::format_descriptor<uint64_t>::format(),
                        shape.ndim(),
                        shape.shape,
                        shape.stride
                    );
                case DTypeEnum::KUInt16:
                    return py::buffer_info(
                        points,
                        dtype.byte_size(),
                        py::format_descriptor<uint16_t>::format(),
                        shape.ndim(),
                        shape.shape,
                        shape.stride
                    );
                case DTypeEnum::KUInt8:
                    return py::buffer_info(
                        points,
                        dtype.byte_size(),
                        py::format_descriptor<uint8_t>::format(),
                        shape.ndim(),
                        shape.shape,
                        shape.stride
                    );
                case DTypeEnum::KFloat32:
                    return py::buffer_info(
                        points,
                        dtype.byte_size(),
                        py::format_descriptor<float_t>::format(),
                        shape.ndim(),
                        shape.shape,
                        shape.stride
                    );
                case DTypeEnum::KFloat64:
                    return py::buffer_info(
                        points,
                        dtype.byte_size(),
                        py::format_descriptor<double_t>::format(),
                        shape.ndim(),
                        shape.shape,
                        shape.stride
                    );
                case DTypeEnum::KBool:
                    return py::buffer_info(
                        points,
                        dtype.byte_size(),
                        py::format_descriptor<bool>::format(),
                        shape.ndim(),
                        shape.shape,
                        shape.stride
                    );
                case DTypeEnum::KUnKnown:
                    return py::buffer_info(
                        points,
                        dtype.byte_size(),
                        py::format_descriptor<void*>::format(),
                        shape.ndim(),
                        shape.shape,
                        shape.stride
                    );
                default:
                    throw std::runtime_error("未知的类型!");
            }
        }

        inline std::string TensorBinding::to_string() const {
            return this->tensor_.get()->to_string();
        }

        inline GenericRawDType TensorBinding::flatten_get(const py::int_ &index) const {
            switch (this->get_dtype().type) {
                case DTypeEnum::KBool:
                    return this->tensor_->ptr<bool>(index);
                case DTypeEnum::KFloat32:
                    return this->tensor_->ptr<float_t>(index);
                case DTypeEnum::KFloat64:
                    return this->tensor_->ptr<double_t>(index);
                case DTypeEnum::KInt8:
                    return this->tensor_->ptr<int8_t>(index);
                case DTypeEnum::KInt16:
                    return this->tensor_->ptr<int16_t>(index);
                case DTypeEnum::KInt32:
                    return this->tensor_->ptr<int32_t>(index);
                case DTypeEnum::KInt64:
                    return this->tensor_->ptr<int64_t>(index);
                case DTypeEnum::KUInt8:
                    return this->tensor_->ptr<uint8_t>(index);
                case DTypeEnum::KUInt16:
                    return this->tensor_->ptr<uint16_t>(index);
                case DTypeEnum::KUInt32:
                    return this->tensor_->ptr<uint32_t>(index);
                case DTypeEnum::KUInt64:
                    return this->tensor_->ptr<uint64_t>(index);
                default:
                    FATAL("暂不支持的DTypeEnum类型: " << static_cast<int>(this->get_dtype().type) << "!");
            }
        }

        inline TensorBinding TensorBinding::get_item(const std::vector<Slice> &slices) const {
            Tensor new_tensor =  this->tensor_.get()->get_item(slices);
            return TensorBinding(new_tensor);
        }


        inline TensorBinding zeros(Shape shape, DType dtype, DeviceEnum device_enum) {
            if (device_enum == DeviceEnum::KCpu) {
                Tensor tensor(HostAllocatorFactory::get_instance(), shape, dtype);
                HostAllocatorFactory::get_instance()->memset_zero(tensor.ptr(), size_t(shape.elem_size() * dtype.byte_size()));
                return TensorBinding(tensor);
            }
            if (device_enum == DeviceEnum::KCuda) {
                Tensor tensor(CudaAllocatorFactory::get_instance(), shape, dtype);
                CudaAllocatorFactory::get_instance()->memset_zero(tensor.ptr(), size_t(shape.elem_size() * dtype.byte_size()));
                return TensorBinding(tensor);
            }
            throw std::runtime_error("未知的设备类型!!!");
        }


        inline TensorBinding from_numpy(const py::array &ndarray) {
            auto buffer_info = ndarray.request();
            auto buffer_ptr = buffer_info.ptr;
            auto old_shape = buffer_info.shape;
            auto shape_vector = std::vector<size_t>();
            for (size_t idx = 0; idx < old_shape.size(); idx++) {
                shape_vector.push_back(old_shape[idx]);
            }
            DTypeEnum dtype_enum;
            switch (ndarray.dtype().kind()) {
                case 'b':  // 布尔类型
                    dtype_enum = DTypeEnum::KBool;
                break;
                case 'i':  // 整数类型
                    switch (ndarray.dtype().itemsize()) {
                        case 1:
                            dtype_enum = DTypeEnum::KInt8;
                        break;
                        case 2:
                            dtype_enum = DTypeEnum::KInt16;
                        break;
                        case 4:
                            dtype_enum = DTypeEnum::KInt32;
                        break;
                        case 8:
                            dtype_enum = DTypeEnum::KInt64;
                        break;
                        default:
                            dtype_enum = DTypeEnum::KUnKnown;
                        break;
                    }
                break;
                case 'u':  // 无符号整数类型
                    switch (ndarray.dtype().itemsize()) {
                        case 1:
                            dtype_enum = DTypeEnum::KUInt8;
                        break;
                        case 2:
                            dtype_enum = DTypeEnum::KUInt16;
                        break;
                        case 4:
                            dtype_enum = DTypeEnum::KUInt32;
                        break;
                        case 8:
                            dtype_enum = DTypeEnum::KUInt64;
                        break;
                        default:
                            dtype_enum = DTypeEnum::KUnKnown;
                        break;
                    }
                break;
                case 'f':  // 浮点数类型
                    switch (ndarray.dtype().itemsize()) {
                        case 4:
                            dtype_enum = DTypeEnum::KFloat32;
                        break;
                        case 8:
                            dtype_enum = DTypeEnum::KFloat64;
                        break;
                        default:
                            dtype_enum = DTypeEnum::KUnKnown;
                        break;
                    }
                break;
                default:
                    dtype_enum = DTypeEnum::KUnKnown;
                break;
            }
            CHECK(dtype_enum != DTypeEnum::KUnKnown);
            const auto dtype = DType(dtype_enum);
            std::shared_ptr<Buffer> buffer = std::make_shared<Buffer>(
                buffer_info.itemsize * buffer_info.size,
                HostAllocatorFactory::get_instance(),
                buffer_ptr,
                false
            );
            return TensorBinding(Tensor(buffer, Shape(shape_vector), dtype));
        }
    }
}

#endif //TENSOR_BINDING_HPP
