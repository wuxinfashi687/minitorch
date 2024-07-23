//
// Created by JUMPWORK-SERVER-01 on 2024/7/15.
//

#ifndef DTYPE_HPP
#define DTYPE_HPP

#include<cstdint>
#include<variant>
#include<pybind11/pybind11.h>

namespace py = pybind11;


namespace minitorch {
    using GenericRawDType = std::variant<
            uint32_t,
            uint64_t,
            uint16_t,
            uint8_t,
            int32_t,
            int64_t,
            int16_t,
            int8_t,
            float_t,
            double_t,
            bool,
            py::none
        >;


    using GenericRawIntType = std::variant<
        int32_t,
        int64_t,
        int16_t,
        int8_t
    >;


    using GenericRawUIntType = std::variant<
        uint32_t,
        uint64_t,
        uint16_t,
        uint8_t
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
        DType(DTypeEnum dtype_enum);
        ~DType() = default;
        DTypeEnum type;
        size_t byte_size() const;
    };

    inline DType::DType(const DTypeEnum dtype_enum) {
        this->type = dtype_enum;
    }

    inline size_t DType::byte_size() const{
        switch (this->type) {
            case DTypeEnum::KInt32:
                return sizeof(int32_t);
            case DTypeEnum::KInt64:
                return sizeof(int64_t);
            case DTypeEnum::KInt16:
                return sizeof(int16_t);
            case DTypeEnum::KInt8:
                return sizeof(int8_t);
            case DTypeEnum::KUInt32:
                return sizeof(uint32_t);
            case DTypeEnum::KUInt64:
                return sizeof(uint64_t);
            case DTypeEnum::KUInt16:
                return sizeof(uint16_t);
            case DTypeEnum::KUInt8:
                return sizeof(uint8_t);
            case DTypeEnum::KFloat32:
                return sizeof(float_t);
            case DTypeEnum::KFloat64:
                return sizeof(double_t);
            case DTypeEnum::KBool:
                return sizeof(bool);
            case DTypeEnum::KUnKnown:
                return sizeof(0);
            default:
                throw std::runtime_error("未知的类型!");
        }
    }

}

#endif //DTYPE_HPP
