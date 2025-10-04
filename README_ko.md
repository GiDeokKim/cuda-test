# CUDA-Vision-Math

[🇺🇸 English Version (영문 README)](README.md)

---

## 🚀 개요 (Overview)

**CUDA-Vision-Math**는 **NVIDIA CUDA 기술**을 사용하여 복잡한 계산 작업을 가속화하기 위해 개발된 고성능 오픈소스 라이브러리입니다.

이 저장소는 주로 다음 분야에 초점을 맞춘 고도로 최적화된 GPU 가속 알고리즘을 제공합니다:

1.  **컴퓨터 비전 (CV) 및 이미지 분석:** 핵심 이미지 처리 연산, 특징 추출 및 조작.

2.  **수치 해석 및 수학:** 선형 대수, 미분 방정식 솔버, 일반적인 수학 루틴.

이 오픈소스 라이브러리의 목표는 개발자와 연구자들이 현대 GPU의 **병렬 처리 능력**을 활용하여 까다로운 애플리케이션의 실행 시간을 획기적으로 줄일 수 있는 툴킷을 제공하는 것입니다.

---

## ✨ 주요 기능 (Features)

- **CUDA 가속:** 모든 핵심 루틴은 최대 병렬 성능을 위해 최적화된 CUDA 커널을 사용하여 구현되었습니다.

- **Bazel 빌드 시스템:** Bazel을 사용하여 빠르고 재현 가능하며 확장성 있는 빌드를 지원하며, 복잡한 CUDA 의존성을 완벽하게 통합합니다.

- **폭넓은 범위:** CV (예: 필터, 변환) 및 고급 수학 (예: FFT, 특수 행렬 연산) 관련 알고리즘을 폭넓게 포함합니다.

- **C++ 인터페이스:** 깔끔하고 통합하기 쉬운 C++ 인터페이스로 설계되었습니다.

---

## 🛠️ 시작하기 (Getting Started)

### 전제 조건 (Prerequisites)

이 라이브러리를 빌드하고 사용하려면 다음이 필요합니다:

- CUDA 기능을 갖춘 NVIDIA GPU (Compute Capability 5.0 이상 권장).

- **CUDA Toolkit** (버전 X.X 이상).

- C++17 이상을 지원하는 C++ 컴파일러.

- **Bazel** (버전 5.0 이상 권장).

### 설치 및 빌드 (Installation and Build)

이 프로젝트는 Bazel을 사용하여 빌드됩니다. 다음 명령어로 모든 타겟을 한 번에 빌드할 수 있습니다:

```bash
# 1. 저장소 클론
git clone https://github.com/YourUsername/cuda-vision-math.git
cd cuda-vision-math

# 2. 모든 라이브러리 및 실행 파일 빌드
bazel build //...

# 선택 사항: 테스트 실행
bazel test //...
```

---

## 💡 사용 예시 (Usage Example - Placeholder)

각 구성 요소(예: `//vision:convolution_lib`)에 대한 구체적인 Bazel 타겟을 포함한 자세한 사용 가이드와 C++ 통합 예시는 여기에 제공될 예정입니다.

```cpp
// 예시: 가속 이미지 필터 (C++ / CUDA)

// #include "cvm/vision/image_filter.h"

// int main() {
//     // 이미지 데이터 로드 (CPU)
//     // ...

//     // GPU로 데이터 전송 및 CUDA 필터 실행
//     // cvm::gpu_image input_gpu = cvm::upload(cpu_data);
//     // cvm::gpu_image output_gpu = cvm::fast_gaussian_blur(input_gpu, 5.0);

//     // 결과 데이터를 다시 CPU로 전송
//     // ...
//     return 0;
// }
```

---

## 📜 라이선스 (License)

이 프로젝트는 **Beerware License** 하에 라이선스되었습니다.

만약 이 라이브러리가 귀하의 작업에 가치가 있거나 **상당한 성능 향상**을 제공했다면, 주저하지 말고 **저자에게 맥주 한 잔을 사주시면 됩니다!** 🍺
