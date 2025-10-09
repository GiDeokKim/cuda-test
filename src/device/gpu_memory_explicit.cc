#include "gpu_memory_template.h"

namespace cuda_vision_math
{
    namespace device
    {

        // 명시적 인스턴스화 - 구체적인 타입들에 대해 템플릿 인스턴스 생성
        // 이렇게 하면 DLL에서 해당 타입들의 템플릿이 미리 컴파일되어 내보내기됨

        // GpuMemory 템플릿 명시적 인스턴스화
        template class GPU_MEMORY_API GpuMemory<int>;
        template class GPU_MEMORY_API GpuMemory<float>;
        template class GPU_MEMORY_API GpuMemory<double>;
        template class GPU_MEMORY_API GpuMemory<unsigned char>;

        // HostMemory 템플릿 명시적 인스턴스화
        template class GPU_MEMORY_API HostMemory<int>;
        template class GPU_MEMORY_API HostMemory<float>;
        template class GPU_MEMORY_API HostMemory<double>;
        template class GPU_MEMORY_API HostMemory<unsigned char>;

    } // namespace device
} // namespace cuda_vision_math
