//
// Created by JUMPWORK-SERVER-01 on 2024/7/8.
//

#ifndef BUFFER_HPP
#define BUFFER_HPP

#include<memory>

#include"base/memory_allocator.hpp"
#include"base/type_extension.hpp"

namespace minitorch {
    class Buffer: public NoCopyable, public std::enable_shared_from_this<Buffer> {
        size_t byte_size_ = 0;
        void* data_ptr_ = nullptr;
        bool use_external_ = false;
        DeviceEnum device_enum_ = DeviceEnum::KUnknown;
        std::shared_ptr<MemoryAllocator> allocator_;
    public:
        explicit Buffer() = default;
        explicit Buffer(
            size_t byte_size,
            std::shared_ptr<MemoryAllocator> allocator = nullptr,
            void* ptr = nullptr,
            bool use_external = false
        );
        ~Buffer();
        bool allocate();
        void copy_from(const Buffer& buffer) const;
        void copy_from(const Buffer* buffer) const;
        void* ptr();
        const void* ptr() const;
        size_t byte_size() const;
        std::shared_ptr<MemoryAllocator> allocator() const;
        DeviceEnum device_type() const;
        void set_device_type(DeviceEnum device_type);
        std::shared_ptr<Buffer> get_shared_from_this();
        bool is_external() const;
    };

    inline Buffer::Buffer(
        size_t byte_size,
        std::shared_ptr<MemoryAllocator> allocator,
        void* ptr,
        bool use_external
    ) {
        this->byte_size_ = byte_size;
        this->allocator_ = allocator;
        this->data_ptr_ = ptr;
        this->use_external_ = use_external;
        if (!this->data_ptr_ && this->allocator_) {
            this->device_enum_ = this->allocator_->get_type();
            this->use_external_ = false;
            this->data_ptr_ = this->allocator_->allocate(byte_size);
        }
    }

    inline Buffer::~Buffer() {
        if (!this->use_external_) {
            if (this->data_ptr_ &&this->allocator_) {
                this->allocator_->release(this->data_ptr_);
                this->data_ptr_ = nullptr;
            }
        }
    }

    inline void* Buffer::ptr() {
        return this->data_ptr_;
    }

    inline const void* Buffer::ptr() const {
        return this->data_ptr_;
    }

    inline size_t Buffer::byte_size() const {
        return this->byte_size_;
    }

    inline void Buffer::copy_from(const Buffer& buffer) const {
        CHECK(this->allocator_ != nullptr);
        CHECK(buffer.data_ptr_ != nullptr);

        size_t byte_size = this->byte_size_ < buffer.byte_size_ ? this->byte_size_ : buffer.byte_size_;
        const DeviceEnum& buffer_device = buffer.device_type();
        const DeviceEnum& current_device = this->device_type();
        CHECK(buffer_device != DeviceEnum::KUnknown && current_device != DeviceEnum::KUnknown);

        if (buffer_device == DeviceEnum::KCpu && current_device == DeviceEnum::KCpu) {
            return this->allocator_->memcpy(buffer.ptr(), this->data_ptr_, byte_size);
        } else if (buffer_device == DeviceEnum::KCuda && current_device == DeviceEnum::KCpu) {
            return this->allocator_->memcpy(
                buffer.ptr(),
                this->data_ptr_,
                byte_size,
                CopyKindEnum::Cuda2Cpu
            );
        } else if (buffer_device == DeviceEnum::KCpu && current_device == DeviceEnum::KCuda) {
            return this->allocator_->memcpy(
                buffer.ptr(),
                this->data_ptr_,
                byte_size,
                CopyKindEnum::Cpu2Cuda
            );
        } else {
            return this->allocator_->memcpy(
                buffer.ptr(),
                this->data_ptr_,
                byte_size,
                CopyKindEnum::Cuda2Cuda
            );
        }
    }

    inline void Buffer::copy_from(const Buffer* buffer) const {
        CHECK(this->allocator_ != nullptr);
        CHECK(buffer != nullptr || buffer->data_ptr_ != nullptr);

        size_t src_size = this->byte_size_;
        size_t dest_size = buffer->byte_size_;
        size_t byte_size = src_size < dest_size ? src_size : dest_size;

        const DeviceEnum& buffer_device = buffer->device_type();
        const DeviceEnum& current_device = this->device_type();
        CHECK(
            buffer_device != DeviceEnum::KUnknown &&
            current_device != DeviceEnum::KUnknown
        );

        if (buffer_device == DeviceEnum::KCpu && current_device == DeviceEnum::KCpu) {
            return this->allocator_->memcpy(buffer->data_ptr_, this->data_ptr_, byte_size);
        } else if (buffer_device == DeviceEnum::KCuda && current_device == DeviceEnum::KCpu) {
            return this->allocator_->memcpy(
                buffer->data_ptr_,
                this->data_ptr_,
                byte_size,
                CopyKindEnum::Cuda2Cpu
            );
        } else if (buffer_device == DeviceEnum::KCpu && current_device == DeviceEnum::KCuda) {
            return this->allocator_->memcpy(
                buffer->data_ptr_,
                this->data_ptr_,
                byte_size,
                CopyKindEnum::Cpu2Cuda
        );
        } else {
            return this->allocator_->memcpy(
                buffer->data_ptr_,
                this->data_ptr_,
                byte_size,
                CopyKindEnum::Cuda2Cuda
            );
        }
    }

    inline DeviceEnum Buffer::device_type() const {
        return this->device_enum_;
    }

    inline void Buffer::set_device_type(const DeviceEnum device_type) {
        this->device_enum_ = device_type;
    }

    inline std::shared_ptr<Buffer> Buffer::get_shared_from_this() {
      return shared_from_this();
    }

    inline bool Buffer::is_external() const {
      return this->use_external_;
    }

}

#endif //BUFFER_HPP
