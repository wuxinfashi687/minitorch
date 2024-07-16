#ifndef DTYPE_HPP
#define DTYPE_HPP

#include<cstdint>
#include<variant>
#include<pybind11/pybind11.h>

namespace py = pybind11;


namespace minitorch
{
    using GenericRawDType = std::variant<
        uint_fast32_t,
        uint_fast64_t,
        uint_fast16_t,
        uint_fast8_t,
        int_fast32_t,
        int_fast64_t,
        int_fast16_t,
        int_fast8_t,
        float_t,
        double_t,
        bool,
        py::none
    >;


    using GenericRawIntType = std::variant<
        int_fast32_t,
        int_fast64_t,
        int_fast16_t,
        int_fast8_t
    >;


    using GenericRawUIntType = std::variant<
        uint_fast32_t,
        uint_fast64_t,
        uint_fast16_t,
        uint_fast8_t
    >;


    using GenericRawFloatType = std::variant<
        float_t,
        double_t
    >;


    enum class DTypeEnum {
        KUnKnown,
        KInt32,
        KInt64,
        KInt16,
        KInt8,
        KUInt32,
        KUInt64,
        KUInt16,
        KUInt8,
        KFloat32,
        KFloat64,
        KBool
    };


    enum class IntEnum {
        KInt32,
        KInt64,
        KInt16,
        KInt8
    };


    enum class UIntEnum {
        KUInt32,
        KUInt64,
        KUInt16,
        KUInt8
    };


    enum class FloatEnum {
        KFloat32,
        KFloat64
    };


    class DType {
    public:
        DType() = default;
        ~DType() = default;
        size_t byte_size;
        using type = GenericRawDType;
        std::string name = "GenericRawDType";
    };


    class KIntType: public DType {
    public:
        using type = GenericRawIntType;
    };


    class KUIntType: public DType {
    public:
        using type = GenericRawUIntType;
    };


    class KFloatType: public DType {
        using type = GenericRawFloatType;
    };


    class KInt32: public KIntType {
    public:
        KInt32() {this->byte_size = sizeof(int_fast32_t);}
        using type = int_fast32_t;
        std::string name = "int_fast32_t";
    };


    class KInt64: public KIntType {
    public:
        KInt64() {this->byte_size = sizeof(int_fast64_t);}
        using type = int_fast64_t;
    };


    class KInt16: public KIntType {
    public:
        KInt16() {this->byte_size = sizeof(int_fast16_t);}
        using type = int_fast16_t;
    };


    class KInt8: public KIntType {
    public:
        KInt8() {this->byte_size = sizeof(int_fast8_t);}
        using type = int_fast8_t;
    };


    class KUInt32: public KUIntType {
    public:
        KUInt32() {this->byte_size = sizeof(uint_fast32_t);}
        using type = uint_fast32_t;
    };


    class KUInt64: public KUIntType {
    public:
        KUInt64() {this->byte_size = sizeof(uint_fast64_t);}
        using type = uint_fast64_t;
    };


    class KUInt16: public KUIntType {
    public:
        KUInt16() {this->byte_size = sizeof(uint_fast16_t);}
        using type = uint_fast16_t;
    };


    class KUInt8: public KUIntType {
    public:
        KUInt8() {this->byte_size = sizeof(uint_fast8_t);}
        using type = uint_fast8_t;
    };


    class KFloat32: public KFloatType {
    public:
        KFloat32() {this->byte_size = sizeof(float);}
        using type = float_t;
    };


    class KFloat64: public KFloatType {
    public:
        KFloat64() {this->byte_size = sizeof(double);}
        using type = double_t;
    };


    class KBool: public DType {
    public:
        KBool() {this->byte_size = sizeof(bool);}
        using type = bool;
    };


    class KUnKnown: public DType {
    public:
        KUnKnown() {this->byte_size = 0;}
        using type = py::none;
    };


#define TYPE_NAME(T) typeid(T).name()
}

#endif
