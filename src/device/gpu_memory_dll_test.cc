#include "gpu_memory.h"

#include <gtest/gtest.h>
#include <vector>
#include <cstring>
#include <chrono>

namespace cuda_vision_math
{
    namespace device
    {

        class GpuMemoryDllTest : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                // CUDA 디바이스 초기화
                cudaError_t error = cudaSetDevice(0);
                ASSERT_EQ(cudaSuccess, error) << "CUDA device setup failed";
            }

            void TearDown() override
            {
                // 디바이스 리셋
                cudaDeviceReset();
            }
        };

        // GpuMemory<int> 기본 기능 테스트
        TEST_F(GpuMemoryDllTest, GpuMemoryIntBasicAllocation)
        {
            GpuMemory<int> memory;
            EXPECT_TRUE(memory.empty());
            EXPECT_EQ(0, memory.size());
            EXPECT_EQ(nullptr, memory.get());

            memory.Allocate(100);
            EXPECT_FALSE(memory.empty());
            EXPECT_EQ(100, memory.size());
            EXPECT_NE(nullptr, memory.get());
        }

        // GpuMemory<int> 생성자 테스트
        TEST_F(GpuMemoryDllTest, GpuMemoryIntConstructorAllocation)
        {
            GpuMemory<int> memory(50);
            EXPECT_FALSE(memory.empty());
            EXPECT_EQ(50, memory.size());
            EXPECT_NE(nullptr, memory.get());
        }

        // GpuMemory<int> 이동 생성자 테스트
        TEST_F(GpuMemoryDllTest, GpuMemoryIntMoveConstructor)
        {
            GpuMemory<int> memory1(100);
            int *original_ptr = memory1.get();
            size_t original_size = memory1.size();

            GpuMemory<int> memory2 = std::move(memory1);

            EXPECT_EQ(original_ptr, memory2.get());
            EXPECT_EQ(original_size, memory2.size());
            EXPECT_TRUE(memory1.empty());
            EXPECT_EQ(nullptr, memory1.get());
        }

        // GpuMemory<int> 이동 할당 연산자 테스트
        TEST_F(GpuMemoryDllTest, GpuMemoryIntMoveAssignment)
        {
            GpuMemory<int> memory1(100);
            GpuMemory<int> memory2(50);

            int *original_ptr = memory1.get();
            size_t original_size = memory1.size();

            memory2 = std::move(memory1);

            EXPECT_EQ(original_ptr, memory2.get());
            EXPECT_EQ(original_size, memory2.size());
            EXPECT_TRUE(memory1.empty());
        }

        // GpuMemory<int> 복사 테스트
        TEST_F(GpuMemoryDllTest, GpuMemoryIntCopyOperations)
        {
            const size_t count = 100;
            GpuMemory<int> device_memory(count);
            std::vector<int> host_memory(count);

            // 호스트 메모리 초기화
            for (size_t i = 0; i < count; ++i)
            {
                host_memory[i] = static_cast<int>(i);
            }

            // 호스트에서 디바이스로 복사
            device_memory.CopyFromHost(host_memory.data(), count);

            // 디바이스에서 호스트로 복사
            std::vector<int> result_memory(count);
            device_memory.CopyToHost(result_memory.data(), count);

            // 결과 검증
            for (size_t i = 0; i < count; ++i)
            {
                EXPECT_EQ(static_cast<int>(i), result_memory[i]);
            }
        }

        // GpuMemory<int> 디바이스 간 복사 테스트
        TEST_F(GpuMemoryDllTest, GpuMemoryIntDeviceToDeviceCopy)
        {
            const size_t count = 50;
            GpuMemory<int> memory1(count);
            GpuMemory<int> memory2(count);

            // 첫 번째 디바이스 메모리 초기화
            std::vector<int> host_memory(count);
            for (size_t i = 0; i < count; ++i)
            {
                host_memory[i] = static_cast<int>(i * 2);
            }
            memory1.CopyFromHost(host_memory.data(), count);

            // 디바이스 간 복사
            memory2.CopyFromDevice(memory1.get(), count);

            // 결과 검증
            std::vector<int> result_memory(count);
            memory2.CopyToHost(result_memory.data(), count);

            for (size_t i = 0; i < count; ++i)
            {
                EXPECT_EQ(static_cast<int>(i * 2), result_memory[i]);
            }
        }

        // GpuMemory<int> memset 테스트 (바이트 단위)
        TEST_F(GpuMemoryDllTest, GpuMemoryIntMemset)
        {
            const size_t count = 100;
            GpuMemory<int> memory(count);

            // 메모리를 0으로 설정
            memory.Memset(0);

            // 결과 검증
            std::vector<int> result_memory(count);
            memory.CopyToHost(result_memory.data(), count);

            for (size_t i = 0; i < count; ++i)
            {
                EXPECT_EQ(0, result_memory[i]);
            }
        }

        // GpuMemory<float> 기본 기능 테스트
        TEST_F(GpuMemoryDllTest, GpuMemoryFloatBasicAllocation)
        {
            GpuMemory<float> memory;
            EXPECT_TRUE(memory.empty());
            EXPECT_EQ(0, memory.size());
            EXPECT_EQ(nullptr, memory.get());

            memory.Allocate(100);
            EXPECT_FALSE(memory.empty());
            EXPECT_EQ(100, memory.size());
            EXPECT_NE(nullptr, memory.get());
        }

        // GpuMemory<float> 복사 테스트
        TEST_F(GpuMemoryDllTest, GpuMemoryFloatCopyOperations)
        {
            const size_t count = 100;
            GpuMemory<float> device_memory(count);
            std::vector<float> host_memory(count);

            // 호스트 메모리 초기화
            for (size_t i = 0; i < count; ++i)
            {
                host_memory[i] = static_cast<float>(i) * 0.5f;
            }

            // 호스트에서 디바이스로 복사
            device_memory.CopyFromHost(host_memory.data(), count);

            // 디바이스에서 호스트로 복사
            std::vector<float> result_memory(count);
            device_memory.CopyToHost(result_memory.data(), count);

            // 결과 검증
            for (size_t i = 0; i < count; ++i)
            {
                EXPECT_FLOAT_EQ(static_cast<float>(i) * 0.5f, result_memory[i]);
            }
        }

        // GpuMemory<double> 기본 기능 테스트
        TEST_F(GpuMemoryDllTest, GpuMemoryDoubleBasicAllocation)
        {
            GpuMemory<double> memory;
            EXPECT_TRUE(memory.empty());
            EXPECT_EQ(0, memory.size());
            EXPECT_EQ(nullptr, memory.get());

            memory.Allocate(100);
            EXPECT_FALSE(memory.empty());
            EXPECT_EQ(100, memory.size());
            EXPECT_NE(nullptr, memory.get());
        }

        // GpuMemory<double> 복사 테스트
        TEST_F(GpuMemoryDllTest, GpuMemoryDoubleCopyOperations)
        {
            const size_t count = 100;
            GpuMemory<double> device_memory(count);
            std::vector<double> host_memory(count);

            // 호스트 메모리 초기화
            for (size_t i = 0; i < count; ++i)
            {
                host_memory[i] = static_cast<double>(i) * 0.25;
            }

            // 호스트에서 디바이스로 복사
            device_memory.CopyFromHost(host_memory.data(), count);

            // 디바이스에서 호스트로 복사
            std::vector<double> result_memory(count);
            device_memory.CopyToHost(result_memory.data(), count);

            // 결과 검증
            for (size_t i = 0; i < count; ++i)
            {
                EXPECT_DOUBLE_EQ(static_cast<double>(i) * 0.25, result_memory[i]);
            }
        }

        // 예외 처리 테스트
        TEST_F(GpuMemoryDllTest, ExceptionHandling)
        {
            // 잘못된 크기로 복사 시도
            GpuMemory<int> memory(10);
            std::vector<int> host_memory(5);

            EXPECT_THROW(memory.CopyFromHost(host_memory.data(), 20), GpuMemoryException);
        }

        // 성능 테스트
        TEST_F(GpuMemoryDllTest, PerformanceTest)
        {
            const size_t count = 1000000;
            GpuMemory<int> device_memory(count);
            std::vector<int> host_memory(count);

            // 호스트 메모리 초기화
            for (size_t i = 0; i < count; ++i)
            {
                host_memory[i] = static_cast<int>(i);
            }

            // 복사 성능 측정
            auto start = std::chrono::high_resolution_clock::now();

            device_memory.CopyFromHost(host_memory.data(), count);
            device_memory.CopyToHost(host_memory.data(), count);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            // 복사가 완료되었는지 확인
            EXPECT_LT(duration.count(), 1000000); // 1초 미만
        }

    } // namespace device
} // namespace cuda_vision_math
