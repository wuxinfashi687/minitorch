//
// Created by JUMPWORK-SERVER-01 on 2024/7/9.
//

#ifndef MEMORY_ALLOCATOR_HPP
#define MEMORY_ALLOCATOR_HPP

#include "base/type_extension.hpp"

#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
    #define MINITORCH_HAVE_POSIX_MEMALIGN
#endif

namespace minitorch {
    class MemoryAllocator {
        DeviceEnum device_enum_ = DeviceEnum::KUnknown;
    public:
        explicit MemoryAllocator(DeviceEnum device_enum);
        virtual DeviceEnum get_type() const;
        virtual void release(void* ptr) = 0;
        virtual void* allocate(size_t byte_size) = 0;
        virtual void memcpy(
            const void* src_ptr,
            void* dest_ptr,
            size_t byte_size,
            CopyKindEnum copy_kind_enum = CopyKindEnum::Cpu2Cpu,
            void* stream = nullptr,
            bool need_sync = false
        ) const;
        virtual void memset_zero(
            void* data_ptr,
            size_t byte_size,
            void* stream = nullptr,
            bool need_sync = false
        );
        virtual ~MemoryAllocator() = default;
    };

    inline MemoryAllocator::MemoryAllocator(const DeviceEnum device_enum) {
        this->device_enum_ = device_enum;
    }

    inline DeviceEnum MemoryAllocator::get_type() const {
        return this->device_enum_;
    }

    inline void MemoryAllocator::memcpy(
        const void* src_ptr,
        void* dest_ptr,
        const size_t byte_size,
        CopyKindEnum copy_kind_enum,
        void* stream,
        bool need_sync
    ) const {
        CHECK_NE(src_ptr, nullptr);
        CHECK_NE(dest_ptr, nullptr);

        if (byte_size == 0) {
            WARNING("MemoryAllocator::memcpy函数拷贝的字节数为0!");
            return;
        }

        cudaStream_t cur_stream = nullptr;
        if (stream != nullptr) {
            cur_stream = static_cast<CUstream_st*>(stream);
        }

        switch (copy_kind_enum) {
            case CopyKindEnum::Cpu2Cpu: {
                std::memcpy(dest_ptr, src_ptr, byte_size);  // 标准库直接复制
                break;
            }
            case CopyKindEnum::Cpu2Cuda: {
                if (cur_stream == nullptr) {
                    cudaMemcpy(
                        dest_ptr, src_ptr, byte_size,
                        cudaMemcpyHostToDevice
                    );  // 同步复制
                } else {
                    cudaMemcpyAsync(
                        dest_ptr, src_ptr, byte_size,
                        cudaMemcpyHostToDevice,
                        cur_stream
                    );  // 异步复制
                }
                break;
            }
            case CopyKindEnum::Cuda2Cpu: {
                if (cur_stream == nullptr) {
                    cudaMemcpy(
                        dest_ptr, src_ptr, byte_size,
                        cudaMemcpyDeviceToHost
                    );
                } else {
                    cudaMemcpyAsync(
                        dest_ptr, src_ptr, byte_size,
                        cudaMemcpyDeviceToHost,
                        cur_stream
                    );
                }
                break;
            }
            case CopyKindEnum::Cuda2Cuda: {
                if (cur_stream == nullptr) {
                    cudaMemcpy(
                        dest_ptr, src_ptr, byte_size,
                        cudaMemcpyDeviceToDevice
                    );
                } else {
                    cudaMemcpyAsync(
                        dest_ptr, src_ptr, byte_size,
                        cudaMemcpyDeviceToDevice,
                        cur_stream
                    );
                }
                break;
            }
            default:
                FATAL("未知的CopyKindEnum类型: " << static_cast<int>(copy_kind_enum));
        }

        if (need_sync) {
            cudaDeviceSynchronize();
        }
    }

    inline void MemoryAllocator::memset_zero(
        void* data_ptr,
        size_t byte_size,
        void* stream,
        bool need_sync
    ) {
        CHECK(this->device_enum_ != DeviceEnum::KUnknown);

        switch (this->device_enum_) {
            case DeviceEnum::KCpu: {
                std::memset(data_ptr, 0, byte_size);
                break;
            }
            case DeviceEnum::KCuda: {
                if (stream != nullptr) {
                    const auto cur_stream = static_cast<cudaStream_t>(stream);
                    cudaMemsetAsync(data_ptr, 0, byte_size, cur_stream);
                } else {
                    cudaMemset(data_ptr, 0, byte_size);
                }
                if (need_sync) {
                    cudaDeviceSynchronize();
                }
                break;
            }
            default:
                FATAL("未知的DeviceEnum类型: " << static_cast<int>(this->device_enum_));
        }
    }


    class HostAllocator: public MemoryAllocator {
    public:
        explicit HostAllocator();
        void release(void* ptr) override;
        void* allocate(size_t byte_size) override;
    };

    inline HostAllocator::HostAllocator(): MemoryAllocator(DeviceEnum::KCpu) {}


    inline void* HostAllocator::allocate(size_t byte_size) {
        if (byte_size == 0) {
            return nullptr;
        }

    #ifdef MINITORCH_HAVE_POSIX_MEMALIGN
        void* data_ptr = nullptr;
        const size_t alignment = (byte_size >= size_t(1024)) ? size_t(32) : size_t(16);
        int status = posix_memalign(
            (void**)&data_ptr,
            ((alignment >= sizeof(void*)) ? alignment : sizeof(void*)),
            byte_size
        );
        if (status != 0) {
            return nullptr;
        }
        return data_ptr;
    #else
        void* data_ptr = malloc(byte_size);
        return data_ptr;
    #endif
    }

    inline void HostAllocator::release(void* ptr) {
        if (ptr != nullptr) {
            free(ptr);
        }
    }


    class HostAllocatorFactory {
        static std::shared_ptr<HostAllocator> instance;
    public:
        static std::shared_ptr<HostAllocator> get_instance();
    };

    inline std::shared_ptr<HostAllocator> HostAllocatorFactory::get_instance() {
        if (instance == nullptr) {
            instance = std::make_shared<HostAllocator>();
        }
        return instance;
    }


    class CudaAllocator: public MemoryAllocator {
    public:
        explicit CudaAllocator();
        void release(void* ptr) override;
        void* allocate(size_t byte_size) override;
    };

    inline CudaAllocator::CudaAllocator(): MemoryAllocator(DeviceEnum::KCuda) {}

    inline void* CudaAllocator::allocate(const size_t byte_size) {
        if (byte_size == 0) {
            return nullptr;
        }

        void* data_ptr = nullptr;
        const cudaError_t err = cudaMalloc(&data_ptr, byte_size);
        CHECK_EQ(err, cudaSuccess);
        return data_ptr;
    }

    inline void CudaAllocator::release(void* ptr) {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }


    class CudaAllocatorFactory {
        static std::shared_ptr<CudaAllocator> instance;
    public:
        static std::shared_ptr<CudaAllocator> get_instance();
    };

    inline std::shared_ptr<CudaAllocator> CudaAllocatorFactory::get_instance() {
        if (instance == nullptr) {
            instance = std::make_shared<CudaAllocator>();
        }
        return instance;
    }


    std::shared_ptr<HostAllocator> HostAllocatorFactory::instance = nullptr;
    std::shared_ptr<CudaAllocator> CudaAllocatorFactory::instance = nullptr;

}

#endif //MEMORY_ALLOCATOR_HPP
