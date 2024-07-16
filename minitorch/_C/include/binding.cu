#include<string>

#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<pybind11/buffer_info.h>

#include"python/tensor_binding.hpp"


namespace py = pybind11;
namespace mt = minitorch;

const char* __Version__ = "0.0.1";


std::string get_version() {
    return std::string(__Version__);
}


PYBIND11_MODULE(binding, m) {
    m.def("get_version", &get_version);
    py::class_<mt::Shape>(m, "__Shape")
        .def_readwrite("_shape", &mt::Shape::shape)
        .def(py::init<std::vector<size_t>>(), py::arg("sizes"))
        .def(py::init())
        .def("__getitem__", &mt::Shape::get_item, py::arg("index"))
        .def("__setitem__", &mt::Shape::set_item, py::arg("index"), py::arg("item"))
        .def("_ndim", &mt::Shape::ndim);
    py::class_<mt::GenericRawDType>(m, "__GenericRawDType")
        .def(py::init());
    py::class_<mt::GenericRawIntType>(m, "__GenericRawIntType")
        .def(py::init());
    py::class_<mt::GenericRawFloatType>(m, "__GenericRawFloatType")
        .def(py::init());
    py::enum_<mt::DTypeEnum>(m, "DTypeEnum")
        .value("KUnKnown", mt::DTypeEnum::KUnKnown)
        .value("KInt32", mt::DTypeEnum::KInt32)
        .value("KInt64", mt::DTypeEnum::KInt64)
        .value("KInt16", mt::DTypeEnum::KInt16)
        .value("KInt8", mt::DTypeEnum::KInt8)
        .value("KUInt32", mt::DTypeEnum::KUInt32)
        .value("KUInt64", mt::DTypeEnum::KUInt64)
        .value("KUInt16", mt::DTypeEnum::KUInt16)
        .value("KUInt8", mt::DTypeEnum::KUInt8)
        .value("KFloat32", mt::DTypeEnum::KFloat32)
        .value("KFloat64", mt::DTypeEnum::KFloat64)
        .value("KBool", mt::DTypeEnum::KBool);
    py::enum_<mt::IntEnum>(m, "IntEnum")
        .value("KInt32", mt::IntEnum::KInt32)
        .value("KInt64", mt::IntEnum::KInt64)
        .value("KInt16", mt::IntEnum::KInt16)
        .value("KInt8", mt::IntEnum::KInt8);
    py::enum_<mt::UIntEnum>(m, "UIntEnum")
        .value("UKInt32", mt::UIntEnum::KUInt32)
        .value("UKInt64", mt::UIntEnum::KUInt64)
        .value("UKInt16", mt::UIntEnum::KUInt16)
        .value("UKInt8", mt::UIntEnum::KUInt8);
    py::enum_<mt::FloatEnum>(m, "FloatEnum")
        .value("KFloat32", mt::FloatEnum::KFloat32)
        .value("KFloat64", mt::FloatEnum::KFloat64);
    py::enum_<mt::DeviceEnum>(m, "DeviceEnum")
        .value("KUnknown", mt::DeviceEnum::KUnknown)
        .value("KCpu", mt::DeviceEnum::KCpu)
        .value("kCuda", mt::DeviceEnum::KCuda);
    py::enum_<mt::CopyKindEnum>(m, "CopyKindEnum")
        .value("Cpu2Cpu", mt::CopyKindEnum::Cpu2Cpu)
        .value("Cpu2Cuda", mt::CopyKindEnum::Cpu2Cuda)
        .value("Cuda2Cpu", mt::CopyKindEnum::Cuda2Cpu)
        .value("Cuda2Cuda", mt::CopyKindEnum::Cuda2Cuda);
    py::class_<mt::DType>(m, "__DType")
        .def(py::init<mt::DTypeEnum>(), py::arg("dtype_enum"));
    py::class_<mt::python::TensorBinding>(m, "__Tensor")
        .def(
            py::init<py::buffer, mt::Shape, mt::DType, mt::DeviceEnum>(),
            py::arg("buffer"),
            py::arg("shape"),
            py::arg("dtype"),
            py::arg("device_enum")
        )
        // .def_buffer(
        //     [](mt::python::TensorBinding& cls) -> py::buffer_info {return mt::python::TensorBinding::get_buffer(cls);}
        // )
        .def("get_shape", &mt::python::TensorBinding::get_shape)
        .def("get_dtype", &mt::python::TensorBinding::get_dtype);
}