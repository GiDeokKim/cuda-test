#ifndef CUDA_VISION_MATH_DEVICE_CUDA_STREAM_H_
#define CUDA_VISION_MATH_DEVICE_CUDA_STREAM_H_

#include <cuda_runtime.h>
#include <string>
#include "gpu_memory_exception.h"

namespace cuda_vision_math
{
    namespace device
    {
        // CUDA 스트림 관리 클래스
        class GPU_MEMORY_API CudaStream
        {
        public:
            CudaStream() : stream_(nullptr)
            {
                cudaError_t error = cudaStreamCreate(&stream_);
                if (error != cudaSuccess)
                {
                    throw GpuMemoryException("cudaStreamCreate failed: " +
                                             std::string(cudaGetErrorString(error)));
                }
            }

            // 복사 생성자 (비활성화)
            CudaStream(const CudaStream &) = delete;
            CudaStream &operator=(const CudaStream &) = delete;

            // 이동 생성자
            CudaStream(CudaStream &&other) noexcept : stream_(other.stream_)
            {
                other.stream_ = nullptr;
            }

            // 이동 할당 연산자
            CudaStream &operator=(CudaStream &&other) noexcept
            {
                if (this != &other)
                {
                    Destroy();
                    stream_ = other.stream_;
                    other.stream_ = nullptr;
                }
                return *this;
            }

            // 소멸자
            ~CudaStream() { Destroy(); }

            // 스트림 포인터 반환
            cudaStream_t get() const { return stream_; }

            // 스트림 동기화
            void Synchronize()
            {
                if (stream_ != nullptr)
                {
                    cudaError_t error = cudaStreamSynchronize(stream_);
                    if (error != cudaSuccess)
                    {
                        throw GpuMemoryException("cudaStreamSynchronize failed: " +
                                                 std::string(cudaGetErrorString(error)));
                    }
                }
            }

            // 스트림 유효성 검사
            bool IsValid() const { return stream_ != nullptr; }

            // 스트림 변환 연산자
            operator cudaStream_t() const { return stream_; }

        private:
            void Destroy()
            {
                if (stream_ != nullptr)
                {
                    cudaStreamDestroy(stream_);
                    stream_ = nullptr;
                }
            }

            cudaStream_t stream_;
        };
    }
}

#endif // CUDA_VISION_MATH_DEVICE_CUDA_STREAM_H_
