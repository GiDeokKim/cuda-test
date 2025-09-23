# CUDA-Vision-Math

[🇰🇷 Korean Version (한국어 README)](README_ko.md)

---

## 🚀 Overview

**CUDA-Vision-Math** is a high-performance open-source library developed to accelerate complex computational tasks using **NVIDIA CUDA technology**.

This repository primarily offers highly optimized, GPU-accelerated algorithms focused on the following areas:

1.  **Computer Vision (CV) and Image Analysis:** Core image processing operations, feature extraction, and manipulation.

2.  **Numerical Computing and Mathematics:** Linear algebra, differential equation solvers, and general mathematical routines.

The goal of this open-source library is to provide developers and researchers with a toolkit that can drastically reduce the execution time of demanding applications by leveraging the **parallel processing power of modern GPUs**.

---

## ✨ Features

- **CUDA Acceleration:** All core routines are implemented using optimized CUDA kernels for maximum parallel performance.

- **Bazel Build System:** Supports fast, reproducible, and scalable builds using Bazel, perfectly integrating complex CUDA dependencies.

- **Broad Coverage:** Includes a wide range of algorithms relevant to CV (e.g., filters, transforms) and advanced math (e.g., FFT, specialized matrix operations).

- **C++ Interface:** Designed with a clean and easy-to-integrate C++ interface.

---

## 🛠️ Getting Started

### Prerequisites

To build and use this library, you will need:

- NVIDIA GPU with CUDA capability (Compute Capability 5.0 or higher recommended).

- **CUDA Toolkit** (version X.X or later).

- A C++ compiler supporting C++17 or later.

- **Bazel** (version 5.0 or later recommended).

### Installation and Build

This project is built using Bazel. You can build all targets at once with the following commands:

```bash
# 1. Clone the repository
git clone https://github.com/YourUsername/cuda-vision-math.git
cd cuda-vision-math

# 2. Build all libraries and executables
bazel build //...

# Optional: Run tests
bazel test //...
```

---

## 💡 Usage Example (Placeholder)

Detailed usage guides, including specific Bazel targets for each component (e.g., `//vision:convolution_lib`), and C++ integration examples will be provided here.

```cpp
// Example: Accelerated Image Filter (C++ / CUDA)

// #include "cvm/vision/image_filter.h"

// int main() {
//     // Load image data (CPU)
//     // ...

//     // Transfer data to GPU and run CUDA filter
//     // cvm::gpu_image input_gpu = cvm::upload(cpu_data);
//     // cvm::gpu_image output_gpu = cvm::fast_gaussian_blur(input_gpu, 5.0);

//     // Transfer results back to CPU
//     // ...
//     return 0;
// }
```

---

## 📜 License

This project is licensed under the **Beerware License**.

If this library proves valuable for your work or provides a **significant performance boost**, feel free to **buy the author a beer in return!** 🍺
