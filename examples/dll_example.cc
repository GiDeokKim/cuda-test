#include "src/device/gpu_memory.h"
#include <iostream>
#include <vector>

int main()
{
    try
    {
        std::cout << "=== GPU Memory DLL 사용 예제 ===" << std::endl;

        // GpuMemory<int> 사용 예제
        std::cout << "\n1. GpuMemory<int> 사용:" << std::endl;
        cuda_vision_math::device::GpuMemory<int> int_memory(100);
        std::cout << "메모리 할당 완료: " << int_memory.size() << " 개의 int" << std::endl;

        // 호스트 데이터 준비
        std::vector<int> host_data(100);
        for (int i = 0; i < 100; ++i)
        {
            host_data[i] = i * 2;
        }

        // 호스트에서 디바이스로 복사
        int_memory.CopyFromHost(host_data.data(), 100);
        std::cout << "호스트에서 디바이스로 복사 완료" << std::endl;

        // 디바이스에서 호스트로 복사
        std::vector<int> result_data(100);
        int_memory.CopyToHost(result_data.data(), 100);
        std::cout << "디바이스에서 호스트로 복사 완료" << std::endl;

        // 결과 검증
        bool correct = true;
        for (int i = 0; i < 100; ++i)
        {
            if (result_data[i] != i * 2)
            {
                correct = false;
                break;
            }
        }
        std::cout << "데이터 검증: " << (correct ? "성공" : "실패") << std::endl;

        // GpuMemory<float> 사용 예제
        std::cout << "\n2. GpuMemory<float> 사용:" << std::endl;
        cuda_vision_math::device::GpuMemory<float> float_memory(50);
        std::cout << "메모리 할당 완료: " << float_memory.size() << " 개의 float" << std::endl;

        // 호스트 데이터 준비
        std::vector<float> float_host_data(50);
        for (int i = 0; i < 50; ++i)
        {
            float_host_data[i] = static_cast<float>(i) * 0.5f;
        }

        // 복사 및 검증
        float_memory.CopyFromHost(float_host_data.data(), 50);
        std::vector<float> float_result_data(50);
        float_memory.CopyToHost(float_result_data.data(), 50);

        correct = true;
        for (int i = 0; i < 50; ++i)
        {
            if (std::abs(float_result_data[i] - static_cast<float>(i) * 0.5f) > 1e-6f)
            {
                correct = false;
                break;
            }
        }
        std::cout << "Float 데이터 검증: " << (correct ? "성공" : "실패") << std::endl;

        // GpuMemory<double> 사용 예제
        std::cout << "\n3. GpuMemory<double> 사용:" << std::endl;
        cuda_vision_math::device::GpuMemory<double> double_memory(25);
        std::cout << "메모리 할당 완료: " << double_memory.size() << " 개의 double" << std::endl;

        // 호스트 데이터 준비
        std::vector<double> double_host_data(25);
        for (int i = 0; i < 25; ++i)
        {
            double_host_data[i] = static_cast<double>(i) * 0.25;
        }

        // 복사 및 검증
        double_memory.CopyFromHost(double_host_data.data(), 25);
        std::vector<double> double_result_data(25);
        double_memory.CopyToHost(double_result_data.data(), 25);

        correct = true;
        for (int i = 0; i < 25; ++i)
        {
            if (std::abs(double_result_data[i] - static_cast<double>(i) * 0.25) > 1e-9)
            {
                correct = false;
                break;
            }
        }
        std::cout << "Double 데이터 검증: " << (correct ? "성공" : "실패") << std::endl;

        // CudaStream 사용 예제
        std::cout << "\n4. CudaStream 사용:" << std::endl;
        cuda_vision_math::device::CudaStream stream;
        std::cout << "CUDA 스트림 생성 완료" << std::endl;

        stream.Synchronize();
        std::cout << "스트림 동기화 완료" << std::endl;

        std::cout << "\n=== 모든 테스트 완료 ===" << std::endl;
    }
    catch (const cuda_vision_math::device::GpuMemoryException &e)
    {
        std::cerr << "GPU 메모리 오류: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "일반 오류: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
