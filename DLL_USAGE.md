# GPU Memory DLL 사용 가이드

이 문서는 GPU 메모리 관리 라이브러리를 DLL로 사용하는 방법을 설명합니다.

## 빌드된 파일들

- `libgpu_memory.so` (Linux 공유 라이브러리)
- `libgpu_memory.a` (정적 라이브러리)
- `gpu_memory.h` (통합 헤더 파일)
- `gpu_memory_exception.h` (예외 클래스)
- `gpu_memory_template.h` (템플릿 클래스들)
- `gpu_memory_explicit.cc` (명시적 인스턴스화)
- `cuda_stream.h` (CUDA 스트림 클래스)

## 사용 가능한 클래스들

### 템플릿 클래스들 (명시적 인스턴스화로 DLL 내보내기)

1. **GpuMemory<T>**: 템플릿 기반 GPU 메모리 관리
   - `GpuMemory<int>`: int 타입 전용
   - `GpuMemory<float>`: float 타입 전용
   - `GpuMemory<double>`: double 타입 전용
   - `GpuMemory<unsigned char>`: unsigned char 타입 전용
2. **HostMemory<T>**: 호스트 메모리 관리
   - `HostMemory<int>`: int 타입 전용
   - `HostMemory<float>`: float 타입 전용
   - `HostMemory<double>`: double 타입 전용
   - `HostMemory<unsigned char>`: unsigned char 타입 전용
3. **CudaStream**: CUDA 스트림 관리

## 사용 방법

### 1. 헤더 파일 포함

```cpp
// 전체 사용 (권장)
#include "gpu_memory.h"

// 또는 특정 부분만 사용
#include "gpu_memory_exception.h"
#include "gpu_memory_template.h"
#include "cuda_stream.h"
```

### 2. 기본 사용 예제

```cpp
#include "gpu_memory.h"
#include <vector>
#include <iostream>

int main() {
    try {
        // GpuMemory<int> 사용
        cuda_vision_math::device::GpuMemory<int> memory(100);

        // 호스트 데이터 준비
        std::vector<int> host_data(100);
        for (int i = 0; i < 100; ++i) {
            host_data[i] = i;
        }

        // 호스트에서 디바이스로 복사
        memory.CopyFromHost(host_data.data(), 100);

        // 디바이스에서 호스트로 복사
        std::vector<int> result_data(100);
        memory.CopyToHost(result_data.data(), 100);

        // 결과 출력
        for (int i = 0; i < 10; ++i) {
            std::cout << result_data[i] << " ";
        }
        std::cout << std::endl;

    } catch (const cuda_vision_math::device::GpuMemoryException& e) {
        std::cerr << "오류: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

### 3. 다른 타입들 사용

```cpp
// Float 타입
cuda_vision_math::device::GpuMemoryFloat float_memory(50);
std::vector<float> float_data(50);
float_memory.CopyFromHost(float_data.data(), 50);

// Double 타입
cuda_vision_math::device::GpuMemoryDouble double_memory(25);
std::vector<double> double_data(25);
double_memory.CopyFromHost(double_data.data(), 25);
```

### 4. CUDA 스트림 사용

```cpp
cuda_vision_math::device::CudaStream stream;
// 스트림 작업...
stream.Synchronize();
```

## 컴파일 및 링크

### Bazel 사용 시

```python
cc_binary(
    name = "my_program",
    srcs = ["my_program.cc"],
    deps = [
        "//src/device:gpu_memory",
    ],
)
```

### CMake 사용 시

```cmake
find_library(GPU_MEMORY_LIB gpu_memory PATHS /path/to/lib)
target_link_libraries(your_target ${GPU_MEMORY_LIB})
target_include_directories(your_target PRIVATE /path/to/headers)
```

### 명령줄 컴파일 시

```bash
g++ -o my_program my_program.cc -I/path/to/headers -L/path/to/lib -lgpu_memory -lcudart
```

## 파일 구조

```
src/device/
├── gpu_memory.h                    # 통합 헤더 파일
├── gpu_memory_exception.h          # 예외 클래스
├── gpu_memory_template.h           # 템플릿 클래스들
├── gpu_memory_explicit.cc          # 명시적 인스턴스화
├── cuda_stream.h                   # CUDA 스트림 클래스
└── BUILD.bazel                     # 빌드 설정
```

## 주요 기능들

### 메모리 관리

- `Allocate(size_t count)`: 메모리 할당
- `Free()`: 메모리 해제
- `get()`: 포인터 반환
- `data()`: 포인터 반환
- `size()`: 요소 개수 반환
- `size_bytes()`: 바이트 크기 반환
- `empty()`: 비어있는지 확인

### 데이터 복사

- `CopyFromHost(const T* host_ptr, size_t count)`: 호스트에서 디바이스로
- `CopyToHost(T* host_ptr, size_t count)`: 디바이스에서 호스트로
- `CopyFromDevice(const T* device_ptr, size_t count)`: 디바이스 간 복사

### 기타 기능

- `Memset(int value)`: 메모리 초기화
- `IsValid()`: 유효성 확인
- `operator T*()`: 포인터 변환

## 예외 처리

모든 CUDA 관련 오류는 `GpuMemoryException`으로 던져집니다:

```cpp
try {
    cuda_vision_math::device::GpuMemory<int> memory(100);
    // 작업...
} catch (const cuda_vision_math::device::GpuMemoryException& e) {
    std::cerr << "GPU 메모리 오류: " << e.what() << std::endl;
}
```

## 주의사항

1. **CUDA 런타임 의존성**: CUDA 런타임 라이브러리가 필요합니다
2. **디바이스 초기화**: 사용 전에 CUDA 디바이스가 초기화되어 있어야 합니다
3. **메모리 크기**: 복사 시 할당된 크기를 초과하지 않도록 주의하세요
4. **예외 안전성**: RAII 패턴으로 자동 메모리 해제가 보장됩니다
5. **헤더 파일**: 필요한 헤더만 포함하여 컴파일 시간을 단축할 수 있습니다
6. **명시적 인스턴스화**: 템플릿 클래스들이 명시적 인스턴스화를 통해 DLL로 내보내기됩니다

## 성능 팁

1. **대용량 데이터**: 큰 데이터는 여러 번에 나누어 복사하는 것보다 한 번에 복사하는 것이 효율적입니다
2. **스트림 사용**: 비동기 작업을 위해 CUDA 스트림을 활용하세요
3. **메모리 재사용**: 가능한 경우 메모리를 재할당하지 말고 재사용하세요
4. **헤더 최적화**: 필요한 헤더만 포함하여 컴파일 시간을 단축하세요
5. **템플릿 사용**: 명시적 인스턴스화를 통해 템플릿의 장점을 유지하면서 DLL 내보내기가 가능합니다

## 문제 해결

### 일반적인 오류들

1. **CUDA 초기화 실패**: CUDA 디바이스가 사용 가능한지 확인
2. **메모리 부족**: GPU 메모리 용량 확인
3. **복사 크기 오류**: 할당된 크기와 복사 크기 일치 확인
4. **헤더 파일 누락**: 필요한 헤더 파일이 포함되었는지 확인
5. **DLL 링크 오류**: 라이브러리 경로와 링크 설정 확인

### 디버깅

```cpp
// 메모리 상태 확인
if (!memory.IsValid()) {
    std::cout << "메모리가 유효하지 않습니다" << std::endl;
}

// 크기 확인
std::cout << "할당된 크기: " << memory.size() << std::endl;
std::cout << "바이트 크기: " << memory.size_bytes() << std::endl;
```
