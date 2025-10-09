#include "src/device/gpu_memory.h"

#include <gtest/gtest.h>
#include <vector>
#include <cstring>
#include <chrono>

namespace cuda_vision_math
{
    namespace device
    {

        class GpuMemoryTest : public ::testing::Test
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

        // GpuMemory 기본 기능 테스트
        TEST_F(GpuMemoryTest, BasicAllocation)
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

        // GpuMemory 생성자 테스트
        TEST_F(GpuMemoryTest, ConstructorAllocation)
        {
            GpuMemory<int> memory(50);
            EXPECT_FALSE(memory.empty());
            EXPECT_EQ(50, memory.size());
            EXPECT_NE(nullptr, memory.get());
        }

        // GpuMemory 이동 생성자 테스트
        TEST_F(GpuMemoryTest, MoveConstructor)
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

        // GpuMemory 이동 할당 연산자 테스트
        TEST_F(GpuMemoryTest, MoveAssignment)
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

        // GpuMemory 복사 테스트
        TEST_F(GpuMemoryTest, CopyOperations)
        {
            const size_t count = 100;
            GpuMemory<int> device_memory(count);
            HostMemory<int> host_memory(count);

            // 호스트 메모리 초기화
            for (size_t i = 0; i < count; ++i)
            {
                host_memory.data()[i] = static_cast<int>(i);
            }

            // 호스트에서 디바이스로 복사
            device_memory.CopyFromHost(host_memory.data(), count);

            // 디바이스에서 호스트로 복사
            HostMemory<int> result_memory(count);
            device_memory.CopyToHost(result_memory.data(), count);

            // 결과 검증
            for (size_t i = 0; i < count; ++i)
            {
                EXPECT_EQ(static_cast<int>(i), result_memory.data()[i]);
            }
        }

        // GpuMemory 디바이스 간 복사 테스트
        TEST_F(GpuMemoryTest, DeviceToDeviceCopy)
        {
            const size_t count = 50;
            GpuMemory<int> memory1(count);
            GpuMemory<int> memory2(count);

            // 첫 번째 디바이스 메모리 초기화
            HostMemory<int> host_memory(count);
            for (size_t i = 0; i < count; ++i)
            {
                host_memory.data()[i] = static_cast<int>(i * 2);
            }
            memory1.CopyFromHost(host_memory.data(), count);

            // 디바이스 간 복사
            memory2.CopyFromDevice(memory1.get(), count);

            // 결과 검증
            HostMemory<int> result_memory(count);
            memory2.CopyToHost(result_memory.data(), count);

            for (size_t i = 0; i < count; ++i)
            {
                EXPECT_EQ(static_cast<int>(i * 2), result_memory.data()[i]);
            }
        }

        // GpuMemory memset 테스트
        TEST_F(GpuMemoryTest, Memset)
        {
            const size_t count = 100;
            GpuMemory<unsigned char> memory(count);

            // 메모리를 42로 설정
            memory.Memset(42);

            // 결과 검증
            HostMemory<unsigned char> result_memory(count);
            memory.CopyToHost(result_memory.data(), count);

            for (size_t i = 0; i < count; ++i)
            {
                EXPECT_EQ(42, result_memory.data()[i]);
            }
        }

        // HostMemory 기본 기능 테스트
        TEST_F(GpuMemoryTest, HostMemoryBasicAllocation)
        {
            HostMemory<int> memory;
            EXPECT_TRUE(memory.empty());
            EXPECT_EQ(0, memory.size());
            EXPECT_EQ(nullptr, memory.get());

            memory.Allocate(100);
            EXPECT_FALSE(memory.empty());
            EXPECT_EQ(100, memory.size());
            EXPECT_NE(nullptr, memory.get());
        }

        // HostMemory 생성자 테스트
        TEST_F(GpuMemoryTest, HostMemoryConstructorAllocation)
        {
            HostMemory<int> memory(50);
            EXPECT_FALSE(memory.empty());
            EXPECT_EQ(50, memory.size());
            EXPECT_NE(nullptr, memory.get());
        }

        // HostMemory 이동 생성자 테스트
        TEST_F(GpuMemoryTest, HostMemoryMoveConstructor)
        {
            HostMemory<int> memory1(100);
            int *original_ptr = memory1.get();
            size_t original_size = memory1.size();

            HostMemory<int> memory2 = std::move(memory1);

            EXPECT_EQ(original_ptr, memory2.get());
            EXPECT_EQ(original_size, memory2.size());
            EXPECT_TRUE(memory1.empty());
            EXPECT_EQ(nullptr, memory1.get());
        }

        // HostMemory 이동 할당 연산자 테스트
        TEST_F(GpuMemoryTest, HostMemoryMoveAssignment)
        {
            HostMemory<int> memory1(100);
            HostMemory<int> memory2(50);

            int *original_ptr = memory1.get();
            size_t original_size = memory1.size();

            memory2 = std::move(memory1);

            EXPECT_EQ(original_ptr, memory2.get());
            EXPECT_EQ(original_size, memory2.size());
            EXPECT_TRUE(memory1.empty());
        }

        // HostMemory memset 테스트
        TEST_F(GpuMemoryTest, HostMemoryMemset)
        {
            const size_t count = 100;
            HostMemory<unsigned char> memory(count);

            // 메모리를 42로 설정
            memory.Memset(42);

            // 결과 검증
            for (size_t i = 0; i < count; ++i)
            {
                EXPECT_EQ(42, memory.data()[i]);
            }
        }

        // CudaStream 기본 기능 테스트
        TEST_F(GpuMemoryTest, CudaStreamBasicFunctionality)
        {
            CudaStream stream;
            EXPECT_NE(nullptr, stream.get());

            // 스트림 동기화 테스트
            EXPECT_NO_THROW(stream.Synchronize());
        }

        // CudaStream 이동 생성자 테스트
        TEST_F(GpuMemoryTest, CudaStreamMoveConstructor)
        {
            CudaStream stream1;
            cudaStream_t original_stream = stream1.get();

            CudaStream stream2 = std::move(stream1);

            EXPECT_EQ(original_stream, stream2.get());
            EXPECT_EQ(nullptr, stream1.get());
        }

        // CudaStream 이동 할당 연산자 테스트
        TEST_F(GpuMemoryTest, CudaStreamMoveAssignment)
        {
            CudaStream stream1;
            CudaStream stream2;

            cudaStream_t original_stream = stream1.get();

            stream2 = std::move(stream1);

            EXPECT_EQ(original_stream, stream2.get());
            EXPECT_EQ(nullptr, stream1.get());
        }

        // 예외 처리 테스트
        TEST_F(GpuMemoryTest, ExceptionHandling)
        {
            // 잘못된 크기로 복사 시도
            GpuMemory<int> memory(10);
            HostMemory<int> host_memory(5);

            EXPECT_THROW(memory.CopyFromHost(host_memory.data(), 20), GpuMemoryException);
        }

        // 성능 테스트 (선택적)
        TEST_F(GpuMemoryTest, PerformanceTest)
        {
            const size_t count = 1000000;
            GpuMemory<int> device_memory(count);
            HostMemory<int> host_memory(count);

            // 호스트 메모리 초기화
            for (size_t i = 0; i < count; ++i)
            {
                host_memory.data()[i] = static_cast<int>(i);
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
