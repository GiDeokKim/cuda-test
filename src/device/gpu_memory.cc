#include "gpu_memory.h"

namespace cuda_vision_math
{
    namespace device
    {

        // GpuMemoryInt 구현
        GpuMemoryInt::GpuMemoryInt() : ptr_(nullptr), size_(0) {}

        GpuMemoryInt::GpuMemoryInt(size_t count) : ptr_(nullptr), size_(count)
        {
            if (count > 0)
            {
                Allocate(count);
            }
        }

        GpuMemoryInt::GpuMemoryInt(GpuMemoryInt &&other) noexcept
            : ptr_(other.ptr_), size_(other.size_)
        {
            other.ptr_ = nullptr;
            other.size_ = 0;
        }

        GpuMemoryInt &GpuMemoryInt::operator=(GpuMemoryInt &&other) noexcept
        {
            if (this != &other)
            {
                Free();
                ptr_ = other.ptr_;
                size_ = other.size_;
                other.ptr_ = nullptr;
                other.size_ = 0;
            }
            return *this;
        }

        GpuMemoryInt::~GpuMemoryInt()
        {
            Free();
        }

        void GpuMemoryInt::Allocate(size_t count)
        {
            Free();
            if (count > 0)
            {
                cudaError_t error = cudaMalloc(&ptr_, count * sizeof(int));
                if (error != cudaSuccess)
                {
                    throw GpuMemoryException("cudaMalloc failed: " +
                                             std::string(cudaGetErrorString(error)));
                }
                size_ = count;
            }
        }

        void GpuMemoryInt::Free()
        {
            if (ptr_ != nullptr)
            {
                cudaFree(ptr_);
                ptr_ = nullptr;
                size_ = 0;
            }
        }

        int *GpuMemoryInt::get() const { return ptr_; }
        int *GpuMemoryInt::data() const { return ptr_; }
        size_t GpuMemoryInt::size() const { return size_; }
        size_t GpuMemoryInt::size_bytes() const { return size_ * sizeof(int); }
        bool GpuMemoryInt::empty() const { return size_ == 0; }

        void GpuMemoryInt::CopyFromHost(const int *host_ptr, size_t count)
        {
            if (count > size_)
            {
                throw GpuMemoryException("Copy count exceeds allocated size");
            }
            cudaError_t error = cudaMemcpy(ptr_, host_ptr, count * sizeof(int),
                                           cudaMemcpyHostToDevice);
            if (error != cudaSuccess)
            {
                throw GpuMemoryException("cudaMemcpy H2D failed: " +
                                         std::string(cudaGetErrorString(error)));
            }
        }

        void GpuMemoryInt::CopyToHost(int *host_ptr, size_t count) const
        {
            if (count > size_)
            {
                throw GpuMemoryException("Copy count exceeds allocated size");
            }
            cudaError_t error = cudaMemcpy(host_ptr, ptr_, count * sizeof(int),
                                           cudaMemcpyDeviceToHost);
            if (error != cudaSuccess)
            {
                throw GpuMemoryException("cudaMemcpy D2H failed: " +
                                         std::string(cudaGetErrorString(error)));
            }
        }

        void GpuMemoryInt::CopyFromDevice(const int *device_ptr, size_t count)
        {
            if (count > size_)
            {
                throw GpuMemoryException("Copy count exceeds allocated size");
            }
            cudaError_t error = cudaMemcpy(ptr_, device_ptr, count * sizeof(int),
                                           cudaMemcpyDeviceToDevice);
            if (error != cudaSuccess)
            {
                throw GpuMemoryException("cudaMemcpy D2D failed: " +
                                         std::string(cudaGetErrorString(error)));
            }
        }

        void GpuMemoryInt::Memset(int value)
        {
            cudaError_t error = cudaMemset(ptr_, value, size_bytes());
            if (error != cudaSuccess)
            {
                throw GpuMemoryException("cudaMemset failed: " +
                                         std::string(cudaGetErrorString(error)));
            }
        }

        bool GpuMemoryInt::IsValid() const { return ptr_ != nullptr; }
        GpuMemoryInt::operator int *() const { return ptr_; }

        // GpuMemoryFloat 구현
        GpuMemoryFloat::GpuMemoryFloat() : ptr_(nullptr), size_(0) {}

        GpuMemoryFloat::GpuMemoryFloat(size_t count) : ptr_(nullptr), size_(count)
        {
            if (count > 0)
            {
                Allocate(count);
            }
        }

        GpuMemoryFloat::GpuMemoryFloat(GpuMemoryFloat &&other) noexcept
            : ptr_(other.ptr_), size_(other.size_)
        {
            other.ptr_ = nullptr;
            other.size_ = 0;
        }

        GpuMemoryFloat &GpuMemoryFloat::operator=(GpuMemoryFloat &&other) noexcept
        {
            if (this != &other)
            {
                Free();
                ptr_ = other.ptr_;
                size_ = other.size_;
                other.ptr_ = nullptr;
                other.size_ = 0;
            }
            return *this;
        }

        GpuMemoryFloat::~GpuMemoryFloat()
        {
            Free();
        }

        void GpuMemoryFloat::Allocate(size_t count)
        {
            Free();
            if (count > 0)
            {
                cudaError_t error = cudaMalloc(&ptr_, count * sizeof(float));
                if (error != cudaSuccess)
                {
                    throw GpuMemoryException("cudaMalloc failed: " +
                                             std::string(cudaGetErrorString(error)));
                }
                size_ = count;
            }
        }

        void GpuMemoryFloat::Free()
        {
            if (ptr_ != nullptr)
            {
                cudaFree(ptr_);
                ptr_ = nullptr;
                size_ = 0;
            }
        }

        float *GpuMemoryFloat::get() const { return ptr_; }
        float *GpuMemoryFloat::data() const { return ptr_; }
        size_t GpuMemoryFloat::size() const { return size_; }
        size_t GpuMemoryFloat::size_bytes() const { return size_ * sizeof(float); }
        bool GpuMemoryFloat::empty() const { return size_ == 0; }

        void GpuMemoryFloat::CopyFromHost(const float *host_ptr, size_t count)
        {
            if (count > size_)
            {
                throw GpuMemoryException("Copy count exceeds allocated size");
            }
            cudaError_t error = cudaMemcpy(ptr_, host_ptr, count * sizeof(float),
                                           cudaMemcpyHostToDevice);
            if (error != cudaSuccess)
            {
                throw GpuMemoryException("cudaMemcpy H2D failed: " +
                                         std::string(cudaGetErrorString(error)));
            }
        }

        void GpuMemoryFloat::CopyToHost(float *host_ptr, size_t count) const
        {
            if (count > size_)
            {
                throw GpuMemoryException("Copy count exceeds allocated size");
            }
            cudaError_t error = cudaMemcpy(host_ptr, ptr_, count * sizeof(float),
                                           cudaMemcpyDeviceToHost);
            if (error != cudaSuccess)
            {
                throw GpuMemoryException("cudaMemcpy D2H failed: " +
                                         std::string(cudaGetErrorString(error)));
            }
        }

        void GpuMemoryFloat::CopyFromDevice(const float *device_ptr, size_t count)
        {
            if (count > size_)
            {
                throw GpuMemoryException("Copy count exceeds allocated size");
            }
            cudaError_t error = cudaMemcpy(ptr_, device_ptr, count * sizeof(float),
                                           cudaMemcpyDeviceToDevice);
            if (error != cudaSuccess)
            {
                throw GpuMemoryException("cudaMemcpy D2D failed: " +
                                         std::string(cudaGetErrorString(error)));
            }
        }

        void GpuMemoryFloat::Memset(int value)
        {
            cudaError_t error = cudaMemset(ptr_, value, size_bytes());
            if (error != cudaSuccess)
            {
                throw GpuMemoryException("cudaMemset failed: " +
                                         std::string(cudaGetErrorString(error)));
            }
        }

        bool GpuMemoryFloat::IsValid() const { return ptr_ != nullptr; }
        GpuMemoryFloat::operator float *() const { return ptr_; }

        // GpuMemoryDouble 구현
        GpuMemoryDouble::GpuMemoryDouble() : ptr_(nullptr), size_(0) {}

        GpuMemoryDouble::GpuMemoryDouble(size_t count) : ptr_(nullptr), size_(count)
        {
            if (count > 0)
            {
                Allocate(count);
            }
        }

        GpuMemoryDouble::GpuMemoryDouble(GpuMemoryDouble &&other) noexcept
            : ptr_(other.ptr_), size_(other.size_)
        {
            other.ptr_ = nullptr;
            other.size_ = 0;
        }

        GpuMemoryDouble &GpuMemoryDouble::operator=(GpuMemoryDouble &&other) noexcept
        {
            if (this != &other)
            {
                Free();
                ptr_ = other.ptr_;
                size_ = other.size_;
                other.ptr_ = nullptr;
                other.size_ = 0;
            }
            return *this;
        }

        GpuMemoryDouble::~GpuMemoryDouble()
        {
            Free();
        }

        void GpuMemoryDouble::Allocate(size_t count)
        {
            Free();
            if (count > 0)
            {
                cudaError_t error = cudaMalloc(&ptr_, count * sizeof(double));
                if (error != cudaSuccess)
                {
                    throw GpuMemoryException("cudaMalloc failed: " +
                                             std::string(cudaGetErrorString(error)));
                }
                size_ = count;
            }
        }

        void GpuMemoryDouble::Free()
        {
            if (ptr_ != nullptr)
            {
                cudaFree(ptr_);
                ptr_ = nullptr;
                size_ = 0;
            }
        }

        double *GpuMemoryDouble::get() const { return ptr_; }
        double *GpuMemoryDouble::data() const { return ptr_; }
        size_t GpuMemoryDouble::size() const { return size_; }
        size_t GpuMemoryDouble::size_bytes() const { return size_ * sizeof(double); }
        bool GpuMemoryDouble::empty() const { return size_ == 0; }

        void GpuMemoryDouble::CopyFromHost(const double *host_ptr, size_t count)
        {
            if (count > size_)
            {
                throw GpuMemoryException("Copy count exceeds allocated size");
            }
            cudaError_t error = cudaMemcpy(ptr_, host_ptr, count * sizeof(double),
                                           cudaMemcpyHostToDevice);
            if (error != cudaSuccess)
            {
                throw GpuMemoryException("cudaMemcpy H2D failed: " +
                                         std::string(cudaGetErrorString(error)));
            }
        }

        void GpuMemoryDouble::CopyToHost(double *host_ptr, size_t count) const
        {
            if (count > size_)
            {
                throw GpuMemoryException("Copy count exceeds allocated size");
            }
            cudaError_t error = cudaMemcpy(host_ptr, ptr_, count * sizeof(double),
                                           cudaMemcpyDeviceToHost);
            if (error != cudaSuccess)
            {
                throw GpuMemoryException("cudaMemcpy D2H failed: " +
                                         std::string(cudaGetErrorString(error)));
            }
        }

        void GpuMemoryDouble::CopyFromDevice(const double *device_ptr, size_t count)
        {
            if (count > size_)
            {
                throw GpuMemoryException("Copy count exceeds allocated size");
            }
            cudaError_t error = cudaMemcpy(ptr_, device_ptr, count * sizeof(double),
                                           cudaMemcpyDeviceToDevice);
            if (error != cudaSuccess)
            {
                throw GpuMemoryException("cudaMemcpy D2D failed: " +
                                         std::string(cudaGetErrorString(error)));
            }
        }

        void GpuMemoryDouble::Memset(int value)
        {
            cudaError_t error = cudaMemset(ptr_, value, size_bytes());
            if (error != cudaSuccess)
            {
                throw GpuMemoryException("cudaMemset failed: " +
                                         std::string(cudaGetErrorString(error)));
            }
        }

        bool GpuMemoryDouble::IsValid() const { return ptr_ != nullptr; }
        GpuMemoryDouble::operator double *() const { return ptr_; }

    } // namespace device
} // namespace cuda_vision_math
