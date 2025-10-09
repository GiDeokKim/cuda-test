#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "matrix.cuh"

// kernel.cu에서 정의된 함수들을 사용
extern cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

// gtest main 함수를 사용하기 위해 kernel.cu의 main 함수를 막기 위한 매크로
#define KERNEL_CU_MAIN_DISABLED

// 벡터 덧셈 테스트
TEST(VectorAdditionTest, BasicAddition)
{
  const int arraySize = 5;
  const int a[arraySize] = {1, 2, 3, 4, 5};
  const int b[arraySize] = {10, 20, 30, 40, 50};
  int c[arraySize] = {0};

  cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
  EXPECT_EQ(cudaStatus, cudaSuccess);

  // 예상 결과: {11, 22, 33, 44, 55}
  EXPECT_EQ(c[0], 11);
  EXPECT_EQ(c[1], 22);
  EXPECT_EQ(c[2], 33);
  EXPECT_EQ(c[3], 44);
  EXPECT_EQ(c[4], 55);
}

TEST(VectorAdditionTest, ZeroAddition)
{
  const int arraySize = 3;
  const int a[arraySize] = {0, 0, 0};
  const int b[arraySize] = {5, 10, 15};
  int c[arraySize] = {0};

  cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
  EXPECT_EQ(cudaStatus, cudaSuccess);

  EXPECT_EQ(c[0], 5);
  EXPECT_EQ(c[1], 10);
  EXPECT_EQ(c[2], 15);
}

TEST(VectorAdditionTest, NegativeNumbers)
{
  const int arraySize = 4;
  const int a[arraySize] = {-1, -2, -3, -4};
  const int b[arraySize] = {1, 2, 3, 4};
  int c[arraySize] = {0};

  cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
  EXPECT_EQ(cudaStatus, cudaSuccess);

  EXPECT_EQ(c[0], 0);
  EXPECT_EQ(c[1], 0);
  EXPECT_EQ(c[2], 0);
  EXPECT_EQ(c[3], 0);
}

// 행렬 곱셈 테스트
TEST(MatrixMultiplicationTest, BasicMultiplication)
{
  const int width = 2;
  int M[width * width] = {1, 2, 3, 4};
  int N[width * width] = {5, 6, 7, 8};
  int P[width * width] = {0, 0, 0, 0};

  int result = MatrixMultiplication(M, N, P, width);
  EXPECT_EQ(result, 0); // 성공

  // 예상 결과: [[19, 22], [43, 50]]
  // M = [[1, 2], [3, 4]], N = [[5, 6], [7, 8]]
  // P[0] = 1*5 + 2*7 = 19
  // P[1] = 1*6 + 2*8 = 22
  // P[2] = 3*5 + 4*7 = 43
  // P[3] = 3*6 + 4*8 = 50
  EXPECT_EQ(P[0], 19);
  EXPECT_EQ(P[1], 22);
  EXPECT_EQ(P[2], 43);
  EXPECT_EQ(P[3], 50);
}

TEST(MatrixMultiplicationTest, IdentityMatrix)
{
  const int width = 3;
  int M[width * width] = {1, 0, 0, 0, 1, 0, 0, 0, 1}; // 단위 행렬
  int N[width * width] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  int P[width * width] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  int result = MatrixMultiplication(M, N, P, width);
  EXPECT_EQ(result, 0);

  // 단위 행렬과의 곱셈은 원래 행렬과 같아야 함
  for (int i = 0; i < width * width; i++)
  {
    EXPECT_EQ(P[i], N[i]);
  }
}

TEST(MatrixMultiplicationTest, ZeroMatrix)
{
  const int width = 2;
  int M[width * width] = {0, 0, 0, 0}; // 영행렬
  int N[width * width] = {1, 2, 3, 4};
  int P[width * width] = {0, 0, 0, 0};

  int result = MatrixMultiplication(M, N, P, width);
  EXPECT_EQ(result, 0);

  // 영행렬과의 곱셈은 영행렬이어야 함
  for (int i = 0; i < width * width; i++)
  {
    EXPECT_EQ(P[i], 0);
  }
}

TEST(MatrixMultiplicationTest, LargeMatrix)
{
  const int width = 4;
  int M[width * width] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  int N[width * width] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}; // 단위 행렬
  int P[width * width] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  int result = MatrixMultiplication(M, N, P, width);
  EXPECT_EQ(result, 0);

  // 단위 행렬과의 곱셈은 원래 행렬과 같아야 함
  for (int i = 0; i < width * width; i++)
  {
    EXPECT_EQ(P[i], M[i]);
  }
}
