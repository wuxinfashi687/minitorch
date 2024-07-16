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
            Shape get_shape() const;
            DType get_dtype() const;
            static py::buffer_info get_buffer(TensorBinding& cls);
            ~TensorBinding() = default;
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


        inline Shape TensorBinding::get_shape() const {
            return this->tensor_.get()->get_shape();
        }

        inline DType TensorBinding::get_dtype() const {
            return this->tensor_.get()->get_dtype();
        }

        inline py::buffer_info TensorBinding::get_buffer(TensorBinding& cls) {
            const DType dtype = cls.tensor_.get()->get_dtype();
            const Shape shape = cls.tensor_.get()->get_shape();
            void* points = cls.tensor_.get()->ptr();
            switch (dtype.type) {
                case DTypeEnum::KInt32:
                    return py::buffer_info(
                        points,
                        dtype.byte_size(),
                        py::format_descriptor<int_fast32_t>::format(),
                        shape.ndim(),
                        shape.shape,
                        shape.stride
                    );
                case DTypeEnum::KInt64:
                    return py::buffer_info(
                        points,
                        dtype.byte_size(),
                        py::format_descriptor<int_fast64_t>::format(),
                        shape.ndim(),
                        shape.shape,
                        shape.stride
                    );
                case DTypeEnum::KInt16:
                    return py::buffer_info(
                        points,
                        dtype.byte_size(),
                        py::format_descriptor<int_fast16_t>::format(),
                        shape.ndim(),
                        shape.shape,
                        shape.stride
                    );
                case DTypeEnum::KInt8:
                    return py::buffer_info(
                        points,
                        dtype.byte_size(),
                        py::format_descriptor<int_fast8_t>::format(),
                        shape.ndim(),
                        shape.shape,
                        shape.stride
                    );
                case DTypeEnum::KUInt32:
                    return py::buffer_info(
                        points,
                        dtype.byte_size(),
                        py::format_descriptor<uint_fast32_t>::format(),
                        shape.ndim(),
                        shape.shape,
                        shape.stride
                    );
                case DTypeEnum::KUInt64:
                    return py::buffer_info(
                        points,
                        dtype.byte_size(),
                        py::format_descriptor<uint_fast64_t>::format(),
                        shape.ndim(),
                        shape.shape,
                        shape.stride
                    );
                case DTypeEnum::KUInt16:
                    return py::buffer_info(
                        points,
                        dtype.byte_size(),
                        py::format_descriptor<uint_fast16_t>::format(),
                        shape.ndim(),
                        shape.shape,
                        shape.stride
                    );
                case DTypeEnum::KUInt8:
                    return py::buffer_info(
                        points,
                        dtype.byte_size(),
                        py::format_descriptor<uint_fast8_t>::format(),
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


        // TensorBinding from_numpy(py::array ndarray) {
        //
        // }
    }
}

#endif //TENSOR_BINDING_HPP
