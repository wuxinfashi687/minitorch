//
// Created by JUMPWORK-SERVER-01 on 2024/7/10.
//

#ifndef TYPE_EXTENSION_HPP
#define TYPE_EXTENSION_HPP

#include<stdexcept>
#include<iostream>
#include<sstream>


namespace minitorch
{
    enum class DeviceEnum {
        KUnknown,
        KCpu,
        KCuda
    };


    enum class CopyKindEnum {
        Cpu2Cpu,
        Cpu2Cuda,
        Cuda2Cpu,
        Cuda2Cuda
    };


    class NoCopyable {
    protected:
        NoCopyable() = default;
        ~NoCopyable() = default;
        NoCopyable(const NoCopyable&) = delete;
        NoCopyable& operator=(const NoCopyable&) = delete;
    };


#ifndef CHECK_EQ
    #define CHECK_EQ(a, b) do {                                     \
        if (a == b) {                                               \
            std::cerr << "Check failed: " << #a << " == " << #b <<  \
            " (" << a << " vs. " << b << ")" << std::endl;          \
            std::terminate();                                       \
        }                                                           \
        } while (0)
#endif


#ifndef CHECK_NE
    #define CHECK_NE(a, b) do {                                     \
        if (a != b) {                                               \
            std::cerr << "Check failed: " << #a << " != " << #b <<  \
            " (" << a << " vs. " << b << ")" << std::endl;          \
            std::terminate();                                       \
        }                                                           \
    } while (0)
#endif


#define TOSTR(x) #x


#ifndef ENABLE_LOG
    #define ENABLE_LOG
    #define RESET "\033[0m"
    #define BLACK "\033[30m"
    #define RED "\033[31m"
    #define GREEN "\033[32m"
    #define YELLOW "\033[33m"
    #define BLUE "\033[34m"
    #define PURPLE "\033[35m"
    #define CYAN "\033[36m"
    #define WHITE "\033[37m"

    #define INFO(message) do {std::cout<< GREEN << "[INFO] " << message << RESET << std::endl;} while (0)
    #define WARNING(message) do {std::cout<< YELLOW << "[WARNING] " << message << RESET << std::endl;} while (0)
    #define ERROR(message) do {std::cout<< RED << "[ERROR] " << message << RESET << std::endl;} while (0)
    #define FATAL(message) do {                                             \
        std::cout<< PURPLE << "[FATAL] " << message << RESET << std::endl;  \
        std::terminate();                                                   \
    } while (0)
    #ifdef _DEBUG
        #define DEBUG_LOG(message) do {std::cout<< CYAN << "[CYAN] " << message << RESET << std::endl;} while (0)
    #endif
#endif


#ifndef CHECK
    #define CHECK(condition) do {                            \
        if (!(condition)) {                                  \
            FATAL("Check failed: " << #condition);           \
            std::ostringstream error_message;                \
            error_message<< "Check failed: " << #condition;  \
            throw std::runtime_error(error_message.str());   \
        }                                                    \
    } while (0)
#endif

}

#endif //TYPE_EXTENSION_HPP
