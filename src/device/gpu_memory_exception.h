#ifndef CUDA_VISION_MATH_DEVICE_GPU_MEMORY_EXCEPTION_H_
#define CUDA_VISION_MATH_DEVICE_GPU_MEMORY_EXCEPTION_H_

#include <stdexcept>
#include <string>

// DLL 내보내기 매크로
#ifdef _WIN32
#ifdef GPU_MEMORY_EXPORTS
#define GPU_MEMORY_API __declspec(dllexport)
#else
#define GPU_MEMORY_API __declspec(dllimport)
#endif
#else
#define GPU_MEMORY_API
#endif

namespace cuda_vision_math
{
    namespace device
    {
        // GPU 메모리 할당 실패 예외
        class GPU_MEMORY_API GpuMemoryException : public std::runtime_error
        {
        public:
            explicit GpuMemoryException(const std::string &message)
                : std::runtime_error(message) {}
        };
    }
}

#endif // CUDA_VISION_MATH_DEVICE_GPU_MEMORY_EXCEPTION_H_
