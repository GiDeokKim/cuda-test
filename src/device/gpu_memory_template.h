#ifndef CUDA_VISION_MATH_DEVICE_GPU_MEMORY_TEMPLATE_H_
#define CUDA_VISION_MATH_DEVICE_GPU_MEMORY_TEMPLATE_H_

#include <cuda_runtime.h>
#include <memory>
#include <type_traits>
#include <string>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "gpu_memory_exception.h"

namespace cuda_vision_math
{
    namespace device
    {
        // GPU 메모리 관리를 위한 RAII 클래스 (템플릿)
        template <typename T>
        class GPU_MEMORY_API GpuMemory
        {
        public:
            // 기본 생성자
            GpuMemory() : ptr_(nullptr), size_(0) {}

            // 크기 기반 생성자
            explicit GpuMemory(size_t count) : ptr_(nullptr), size_(count)
            {
                if (count > 0)
                {
                    Allocate(count);
                }
            }

            // 복사 생성자 (비활성화)
            GpuMemory(const GpuMemory &) = delete;
            GpuMemory &operator=(const GpuMemory &) = delete;

            // 이동 생성자
            GpuMemory(GpuMemory &&other) noexcept
                : ptr_(other.ptr_), size_(other.size_)
            {
                other.ptr_ = nullptr;
                other.size_ = 0;
            }

            // 이동 할당 연산자
            GpuMemory &operator=(GpuMemory &&other) noexcept
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

            // 소멸자
            ~GpuMemory() { Free(); }

            // 메모리 할당
            void Allocate(size_t count)
            {
                Free(); // 기존 메모리 해제
                if (count > 0)
                {
                    cudaError_t error = cudaMalloc(&ptr_, count * sizeof(T));
                    if (error != cudaSuccess)
                    {
                        throw GpuMemoryException("cudaMalloc failed: " +
                                                 std::string(cudaGetErrorString(error)));
                    }
                    size_ = count;
                }
            }

            // 메모리 해제
            void Free()
            {
                if (ptr_ != nullptr)
                {
                    cudaFree(ptr_);
                    ptr_ = nullptr;
                    size_ = 0;
                }
            }

            // 포인터 반환
            T *get() const { return ptr_; }
            T *data() const { return ptr_; }

            // 크기 정보
            size_t size() const { return size_; }
            size_t size_bytes() const { return size_ * sizeof(T); }
            bool empty() const { return size_ == 0; }

            // 호스트에서 디바이스로 복사
            void CopyFromHost(const T *host_ptr, size_t count)
            {
                if (count > size_)
                {
                    throw GpuMemoryException("Copy count exceeds allocated size");
                }
                cudaError_t error = cudaMemcpy(ptr_, host_ptr, count * sizeof(T),
                                               cudaMemcpyHostToDevice);
                if (error != cudaSuccess)
                {
                    throw GpuMemoryException("cudaMemcpy H2D failed: " +
                                             std::string(cudaGetErrorString(error)));
                }
            }

            // 디바이스에서 호스트로 복사
            void CopyToHost(T *host_ptr, size_t count) const
            {
                if (count > size_)
                {
                    throw GpuMemoryException("Copy count exceeds allocated size");
                }
                cudaError_t error = cudaMemcpy(host_ptr, ptr_, count * sizeof(T),
                                               cudaMemcpyDeviceToHost);
                if (error != cudaSuccess)
                {
                    throw GpuMemoryException("cudaMemcpy D2H failed: " +
                                             std::string(cudaGetErrorString(error)));
                }
            }

            // 디바이스 간 복사
            void CopyFromDevice(const T *device_ptr, size_t count)
            {
                if (count > size_)
                {
                    throw GpuMemoryException("Copy count exceeds allocated size");
                }
                cudaError_t error = cudaMemcpy(ptr_, device_ptr, count * sizeof(T),
                                               cudaMemcpyDeviceToDevice);
                if (error != cudaSuccess)
                {
                    throw GpuMemoryException("cudaMemcpy D2D failed: " +
                                             std::string(cudaGetErrorString(error)));
                }
            }

            // 메모리 초기화
            void Memset(int value)
            {
                cudaError_t error = cudaMemset(ptr_, value, size_bytes());
                if (error != cudaSuccess)
                {
                    throw GpuMemoryException("cudaMemset failed: " +
                                             std::string(cudaGetErrorString(error)));
                }
            }

            // 유효성 검사
            bool IsValid() const { return ptr_ != nullptr; }

            // 포인터 변환 연산자
            operator T *() const { return ptr_; }

        private:
            T *ptr_;
            size_t size_;
        };

        // 호스트 메모리 관리를 위한 RAII 클래스
        template <typename T>
        class GPU_MEMORY_API HostMemory
        {
        public:
            // 기본 생성자
            HostMemory() : ptr_(nullptr), size_(0) {}

            // 크기 기반 생성자
            explicit HostMemory(size_t count) : ptr_(nullptr), size_(count)
            {
                if (count > 0)
                {
                    Allocate(count);
                }
            }

            // 복사 생성자 (비활성화)
            HostMemory(const HostMemory &) = delete;
            HostMemory &operator=(const HostMemory &) = delete;

            // 이동 생성자
            HostMemory(HostMemory &&other) noexcept
                : ptr_(other.ptr_), size_(other.size_)
            {
                other.ptr_ = nullptr;
                other.size_ = 0;
            }

            // 이동 할당 연산자
            HostMemory &operator=(HostMemory &&other) noexcept
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

            // 소멸자
            ~HostMemory() { Free(); }

            // 메모리 할당
            void Allocate(size_t count)
            {
                Free(); // 기존 메모리 해제
                if (count > 0)
                {
                    ptr_ = static_cast<T *>(malloc(count * sizeof(T)));
                    if (ptr_ == nullptr)
                    {
                        throw GpuMemoryException("malloc failed");
                    }
                    size_ = count;
                }
            }

            // 메모리 해제
            void Free()
            {
                if (ptr_ != nullptr)
                {
                    free(ptr_);
                    ptr_ = nullptr;
                    size_ = 0;
                }
            }

            // 포인터 반환
            T *get() const { return ptr_; }
            T *data() const { return ptr_; }

            // 크기 정보
            size_t size() const { return size_; }
            size_t size_bytes() const { return size_ * sizeof(T); }
            bool empty() const { return size_ == 0; }

            // 메모리 초기화
            void Memset(int value)
            {
                memset(ptr_, value, size_bytes());
            }

            // 유효성 검사
            bool IsValid() const { return ptr_ != nullptr; }

            // 포인터 변환 연산자
            operator T *() const { return ptr_; }

        private:
            T *ptr_;
            size_t size_;
        };
    }
}

#endif // CUDA_VISION_MATH_DEVICE_GPU_MEMORY_TEMPLATE_H_
